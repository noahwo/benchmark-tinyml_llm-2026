import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"  # Read a single video file from the given input_path
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"  # Output the video with drawn rectangles, labels, and mAP
confidence_threshold = 0.5

# =========================
# Utility functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        print(f"Warning: Label file not found at {path}. Detections will use class IDs.")
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for i, line in enumerate(lines):
        if ':' in line:
            # Support "index: label" format
            idx_str, name = line.split(':', 1)
            try:
                idx = int(idx_str.strip())
                labels[idx] = name.strip()
            except ValueError:
                labels[i] = line.strip()
        else:
            labels[i] = line.strip()
    return labels

def preprocess_frame(frame_bgr, input_size, input_dtype):
    # Resize and convert color as most TFLite object detection models expect RGB
    h, w = input_size
    resized = cv2.resize(frame_bgr, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        input_data = (rgb.astype(np.float32) / 255.0).astype(np.float32)
    else:
        input_data = rgb.astype(input_dtype)
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def parse_detections(interpreter, output_details):
    boxes = None
    classes = None
    scores = None
    num = None
    for od in output_details:
        output_data = interpreter.get_tensor(od['index'])
        # Typical shapes:
        # boxes: [1, N, 4], classes: [1, N], scores: [1, N], num: [1]
        if output_data.ndim == 3 and output_data.shape[2] == 4:
            boxes = output_data[0]
        elif output_data.ndim == 2:
            # Could be scores or classes
            # Heuristic: values <= 1.0 likely scores; otherwise classes
            arr = output_data[0]
            if arr.dtype in (np.float32, np.float64):
                if np.nanmax(arr) <= 1.0:
                    scores = arr
                else:
                    classes = arr.astype(np.int32)
            else:
                classes = arr.astype(np.int32)
        elif output_data.ndim == 1 and output_data.shape[0] == 1:
            num = int(output_data[0])
    # Fallbacks in case num is None
    if num is None and scores is not None:
        num = scores.shape[0]
    if boxes is None or classes is None or scores is None or num is None:
        raise RuntimeError("Unexpected TFLite detection output format.")
    return boxes, classes, scores, num

def draw_detections(frame, detections, labels, map_score):
    # Draw bounding boxes and labels on frame
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 175, 255), 2)
        label = labels.get(class_id, str(class_id))
        caption = f"{label}: {score:.2f}"
        cv2.putText(frame, caption, (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 175, 255), 2, cv2.LINE_AA)
    # Draw mAP (proxy) on the top-left
    cv2.putText(frame, f"mAP: {map_score:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

def clip_box_coords(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2

class MAPAggregator:
    """
    Since ground truth annotations are not available from the provided inputs,
    we maintain a proxy mAP metric based on the mean of per-class average scores
    for detections above the confidence threshold. This is not a true mAP, but
    provides a simple, trackable score over time for visualization purposes.
    """
    def __init__(self):
        self.class_conf_sum = {}
        self.class_count = {}

    def update(self, class_id, score):
        if class_id not in self.class_conf_sum:
            self.class_conf_sum[class_id] = 0.0
            self.class_count[class_id] = 0
        self.class_conf_sum[class_id] += float(score)
        self.class_count[class_id] += 1

    def compute(self):
        valid_classes = [cid for cid, cnt in self.class_count.items() if cnt > 0]
        if not valid_classes:
            return 0.0
        per_class_avgs = [self.class_conf_sum[cid] / self.class_count[cid] for cid in valid_classes]
        return float(np.mean(per_class_avgs))

# =========================
# Setup
# =========================
def main():
    print("Initializing TFLite interpreter with EdgeTPU...")
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except ValueError as e:
        print("Failed to load EdgeTPU delegate. Ensure the EdgeTPU runtime is installed and the path is correct.")
        raise e

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input tensor details
    input_shape = input_details[0]['shape']  # [1, height, width, 3]
    input_h, input_w = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']

    # Load labels
    labels = load_labels(label_path)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open input video at {input_path}")
        return

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 1e-2:
        in_fps = 30.0  # default fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, in_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for writing at {output_path}")
        cap.release()
        return

    print("Processing video...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Model: {model_path}")
    print(f"Labels: {label_path}")
    print(f"Frame size: {frame_w}x{frame_h} @ {in_fps:.2f} FPS")
    print(f"Confidence threshold: {confidence_threshold}")

    map_aggregator = MAPAggregator()
    frame_index = 0
    inference_times = []

    start_time_total = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        # Preprocess
        input_data = preprocess_frame(frame, (input_h, input_w), input_dtype)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Inference
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        inference_times.append((t1 - t0) * 1000.0)  # ms

        # Parse outputs
        boxes, classes, scores, num = parse_detections(interpreter, output_details)

        # Build detections list: (x1, y1, x2, y2, score, class_id)
        detections = []
        num = min(num, len(scores), len(classes), len(boxes))
        for i in range(num):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            class_id = int(classes[i])
            # boxes are in [ymin, xmin, ymax, xmax] normalized
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * frame_w)
            y1 = int(ymin * frame_h)
            x2 = int(xmax * frame_w)
            y2 = int(ymax * frame_h)
            x1, y1, x2, y2 = clip_box_coords(x1, y1, x2, y2, frame_w, frame_h)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((x1, y1, x2, y2, score, class_id))
            # Update proxy mAP aggregator
            map_aggregator.update(class_id, score)

        # Compute proxy mAP
        map_score = map_aggregator.compute()

        # Draw
        draw_detections(frame, detections, labels, map_score)

        # Write frame
        writer.write(frame)

        # Optional progress update
        if frame_index % 30 == 0:
            elapsed = time.time() - start_time_total
            fps_proc = frame_index / max(1e-6, elapsed)
            print(f"Processed {frame_index}/{total_frames if total_frames>0 else '?'} frames - ~{fps_proc:.2f} FPS (inference {np.mean(inference_times[-30:]):.1f} ms)")

    cap.release()
    writer.release()

    total_elapsed = time.time() - start_time_total
    avg_inf_ms = float(np.mean(inference_times)) if inference_times else 0.0
    print("Processing complete.")
    print(f"Total time: {total_elapsed:.2f} s")
    print(f"Average inference time: {avg_inf_ms:.2f} ms")
    print(f"Estimated proxy mAP over video: {map_aggregator.compute():.3f}")
    print(f"Saved output video with detections and mAP overlay to: {output_path}")

if __name__ == "__main__":
    main()