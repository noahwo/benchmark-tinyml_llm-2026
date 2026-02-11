import os
import time
import numpy as np
import cv2

# TFLite Runtime (EdgeTPU)
from tflite_runtime.interpreter import Interpreter, load_delegate

# ==============================
# Configuration Parameters
# ==============================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# ==============================
# Utilities
# ==============================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print(f"Warning: Label file not found at {path}. Using empty label map.")
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Try to parse "id: name" or "id name" or plain label list
    for idx, line in enumerate(lines):
        if ':' in line:
            left, right = line.split(':', 1)
            try:
                labels[int(left.strip())] = right.strip()
                continue
            except ValueError:
                pass
        parts = line.split()
        if parts and parts[0].isdigit():
            try:
                labels[int(parts[0])] = ' '.join(parts[1:]).strip()
                continue
            except ValueError:
                pass
        # Fallback: assign by enumerated index
        labels[idx] = line
    return labels

def color_for_class(class_id):
    # Deterministic color from class id (avoid importing random)
    r = (37 * (class_id + 1)) % 255
    g = (17 * (class_id + 1)) % 255
    b = (29 * (class_id + 1)) % 255
    # Avoid too-dark colors
    r = 80 + (r % 175)
    g = 80 + (g % 175)
    b = 80 + (b % 175)
    return int(b), int(g), int(r)  # BGR for OpenCV

def ensure_dir_exists(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def get_video_writer(path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))

def preprocess_frame(frame_bgr, input_size):
    ih, iw = input_size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih))
    return np.expand_dims(resized, axis=0)

def find_output_tensors(interpreter):
    # Return indices for boxes, classes, scores, count based on shapes/values
    output_details = interpreter.get_output_details()
    tensors = [interpreter.get_tensor(od['index']) for od in output_details]
    idx_boxes = idx_classes = idx_scores = idx_count = None

    # Identify count: shape length 1 and size 1
    for i, t in enumerate(tensors):
        if t.ndim == 1 and t.shape[0] == 1:
            idx_count = i
            break

    # Identify boxes: shape [1, N, 4]
    for i, t in enumerate(tensors):
        if t.ndim == 3 and t.shape[0] == 1 and t.shape[2] == 4:
            idx_boxes = i
            break

    # Identify classes & scores among remaining 2D [1, N]
    candidates = []
    for i, t in enumerate(tensors):
        if t.ndim == 2 and t.shape[0] == 1:
            candidates.append(i)
    if len(candidates) >= 2:
        a, b = candidates[0], candidates[1]
        max_a = float(np.max(tensors[a])) if tensors[a].size else 0.0
        max_b = float(np.max(tensors[b])) if tensors[b].size else 0.0
        # Scores expected in [0, 1], classes typically > 1
        if max_a <= 1.05 and max_b > 1.05:
            idx_scores, idx_classes = a, b
        elif max_b <= 1.05 and max_a > 1.05:
            idx_scores, idx_classes = b, a
        else:
            # Fallback to assume standard order if ambiguous
            idx_scores, idx_classes = a, b

    # As ultimate fallback, assume standard TFLite SSD order: 0:boxes,1:classes,2:scores,3:num
    if None in (idx_boxes, idx_classes, idx_scores, idx_count):
        # Attempt standard order mapping
        try:
            if len(output_details) >= 4:
                idx_boxes = 0 if idx_boxes is None else idx_boxes
                idx_classes = 1 if idx_classes is None else idx_classes
                idx_scores = 2 if idx_scores is None else idx_scores
                idx_count = 3 if idx_count is None else idx_count
        except Exception:
            pass

    return (output_details[idx_boxes]['index'],
            output_details[idx_classes]['index'],
            output_details[idx_scores]['index'],
            output_details[idx_count]['index'])

# ==============================
# Main Pipeline
# ==============================
def main():
    # 1. Setup: Load interpreter with EdgeTPU, allocate tensors, load labels, open input video
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print("Loading TFLite model with EdgeTPU delegate...")
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    input_height, input_width = input_details['shape'][1], input_details['shape'][2]
    input_dtype = input_details['dtype']
    if input_dtype != np.uint8:
        print("Warning: Model input is not uint8. Proceeding, but EdgeTPU models typically use uint8.")

    labels = load_labels(label_path)
    print(f"Loaded {len(labels)} labels from {label_path}")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 0 or np.isnan(in_fps):
        in_fps = 30.0

    ensure_dir_exists(output_path)
    writer = get_video_writer(output_path, in_w, in_h, in_fps)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for output: {output_path}")

    # Determine output tensor indices reliably once we have a first inference
    # Do a dry-run with a black frame of correct size
    dummy = np.zeros((1, input_height, input_width, 3), dtype=np.uint8)
    interpreter.set_tensor(input_index, dummy)
    interpreter.invoke()
    out_idx_boxes, out_idx_classes, out_idx_scores, out_idx_count = find_output_tensors(interpreter)

    # mAP proxy: running mean of detection confidences over all processed frames/detections
    sum_confidences = 0.0
    count_detections = 0
    frame_index = 0

    print("Processing video...")
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        orig_h, orig_w = frame.shape[:2]

        # 2. Preprocessing
        input_tensor = preprocess_frame(frame, (input_height, input_width))
        if input_dtype == np.float32:
            # Typical normalization if float32 model (not common for EdgeTPU models)
            input_tensor = (input_tensor.astype(np.float32) - 127.5) / 127.5
        else:
            input_tensor = input_tensor.astype(np.uint8)

        # 3. Inference
        t0 = time.time()
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        infer_ms = (time.time() - t0) * 1000.0

        boxes = interpreter.get_tensor(out_idx_boxes)  # [1, num, 4] in [ymin, xmin, ymax, xmax]
        classes = interpreter.get_tensor(out_idx_classes)  # [1, num]
        scores = interpreter.get_tensor(out_idx_scores)  # [1, num]
        count = interpreter.get_tensor(out_idx_count)  # [1]

        if boxes.ndim != 3 or scores.ndim != 2 or classes.ndim != 2:
            # Fallback re-detection of indices if shapes are unexpected
            out_idx_boxes, out_idx_classes, out_idx_scores, out_idx_count = find_output_tensors(interpreter)
            boxes = interpreter.get_tensor(out_idx_boxes)
            classes = interpreter.get_tensor(out_idx_classes)
            scores = interpreter.get_tensor(out_idx_scores)
            count = interpreter.get_tensor(out_idx_count)

        num = int(count[0]) if count.size else boxes.shape[1]
        boxes = boxes[0]
        classes = classes[0]
        scores = scores[0]

        # 4. Output handling: draw detection boxes with labels and update mAP proxy
        for i in range(min(num, boxes.shape[0])):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            # Convert to absolute coordinates
            left = max(0, min(orig_w - 1, int(xmin * orig_w)))
            top = max(0, min(orig_h - 1, int(ymin * orig_h)))
            right = max(0, min(orig_w - 1, int(xmax * orig_w)))
            bottom = max(0, min(orig_h - 1, int(ymax * orig_h)))

            class_id = int(classes[i])
            label = labels.get(class_id, f"id:{class_id}")
            color = color_for_class(class_id)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label_text = f"{label} {score:.2f}"
            # Text background
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (left, top - th - baseline), (left + tw, top), color, -1)
            cv2.putText(frame, label_text, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Update mAP proxy stats
            sum_confidences += score
            count_detections += 1

        # Compute running mAP proxy (mean of detection confidences)
        map_proxy = (sum_confidences / count_detections) if count_detections > 0 else 0.0
        fps_infer = 1000.0 / infer_ms if infer_ms > 0 else 0.0
        hud_text = f"mAP: {map_proxy:.3f}  Inference: {infer_ms:.1f} ms  FPS: {fps_infer:.1f}"
        cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 240), 2, cv2.LINE_AA)

        writer.write(frame)

        # Optional: print progress every 50 frames
        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames - mAP proxy: {map_proxy:.3f}, last inference: {infer_ms:.1f} ms")

    cap.release()
    writer.release()
    total_time = time.time() - t_start
    total_frames = frame_index
    overall_fps = total_frames / total_time if total_time > 0 else 0.0
    final_map_proxy = (sum_confidences / count_detections) if count_detections > 0 else 0.0

    print("Processing complete.")
    print(f"Saved output video to: {output_path}")
    print(f"Frames: {total_frames}, Time: {total_time:.2f}s, Avg FPS: {overall_fps:.2f}")
    print(f"Final mAP (proxy by mean confidence): {final_map_proxy:.4f}")

if __name__ == "__main__":
    main()