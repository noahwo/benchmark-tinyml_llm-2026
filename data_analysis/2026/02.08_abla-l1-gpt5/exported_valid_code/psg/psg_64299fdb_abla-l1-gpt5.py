import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    labels = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"Warning: Could not load label file '{path}': {e}")
    return labels

def ensure_dir_for_file(file_path):
    out_dir = os.path.dirname(file_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

def preprocess_frame(frame_bgr, input_shape, input_dtype):
    # Convert BGR to RGB, resize to model input size, and normalize/cast
    _, in_h, in_w, _ = input_shape  # expecting NHWC
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h))
    if input_dtype == np.float32:
        input_data = (resized.astype(np.float32) / 255.0)[None, ...]
    else:
        input_data = resized[None, ...].astype(input_dtype)
    return input_data

def parse_detections(interpreter, output_details):
    # Try to robustly extract boxes, classes, scores, and num detections
    boxes = None
    classes = None
    scores = None
    num = None

    for det in output_details:
        arr = interpreter.get_tensor(det["index"])
        # Remove batch dimension where applicable
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr[0]
        elif arr.ndim == 2:
            if arr.dtype in (np.float32, np.float64):
                # Heuristic: if max <= 1.0, likely scores
                if np.max(arr) <= 1.0:
                    scores = arr[0]
                else:
                    classes = arr[0].astype(np.int32)
            else:
                classes = arr[0].astype(np.int32)
        elif arr.size == 1:
            try:
                num = int(arr.reshape(-1)[0])
            except Exception:
                pass

    # Fallback to standard SSD TFLite output ordering if needed
    if boxes is None or classes is None or scores is None:
        try:
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            num = int(interpreter.get_tensor(output_details[3]['index']).reshape(-1)[0])
        except Exception:
            pass

    return boxes, classes, scores, num

def draw_detections(frame_bgr, boxes, classes, scores, num, labels, threshold):
    h, w = frame_bgr.shape[:2]
    if boxes is None or classes is None or scores is None:
        return frame_bgr

    count = len(scores) if num is None else min(int(num), len(scores))
    for i in range(count):
        score = float(scores[i])
        if score < threshold:
            continue

        # Boxes are typically [ymin, xmin, ymax, xmax] normalized [0,1]
        y_min, x_min, y_max, x_max = boxes[i]
        x1 = int(max(0, min(w - 1, x_min * w)))
        y1 = int(max(0, min(h - 1, y_min * h)))
        x2 = int(max(0, min(w - 1, x_max * w)))
        y2 = int(max(0, min(h - 1, y_max * h)))

        class_id = int(classes[i]) if i < len(classes) else -1
        label = labels[class_id] if 0 <= class_id < len(labels) else f"id:{class_id}"

        # Draw bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        caption = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_x2 = min(x1 + tw + 6, w - 1)
        text_bg_y2 = min(y1 - 2 + th + 6, h - 1)
        text_bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(frame_bgr, (x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 255, 0), -1)
        cv2.putText(frame_bgr, caption, (x1 + 3, y1 - 4 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame_bgr

def main():
    # 1) Setup: Load labels and initialize TFLite interpreter
    labels = load_labels(label_path)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Expect a single input tensor (NHWC)
    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]  # [1, h, w, 3]
    input_dtype = input_details[0]["dtype"]

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_path}")
        return

    # Read first frame to setup output writer
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Error: Could not read first frame from: {input_path}")
        cap.release()
        return

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback

    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing: {output_path}")
        cap.release()
        return

    frame_count = 0
    t0 = time.time()

    # Process the already-read first frame and then loop
    frames_to_process = [frame]

    while True:
        if not frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            frames_to_process.append(frame)

        frame_bgr = frames_to_process.pop(0)

        # 2) Preprocessing
        input_data = preprocess_frame(frame_bgr, input_shape, input_dtype)

        # 3) Inference
        interpreter.set_tensor(input_index, input_data)
        inf_start = time.time()
        interpreter.invoke()
        inf_time = (time.time() - inf_start)

        # Parse detection outputs
        boxes, classes, scores, num = parse_detections(interpreter, output_details)

        # 4) Output handling: draw and write frame
        annotated = draw_detections(frame_bgr, boxes, classes, scores, num, labels, confidence_threshold)

        # Optionally overlay inference time
        info_text = f"Inference: {inf_time*1000:.1f} ms"
        cv2.putText(annotated, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)

        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s (avg {avg_fps:.2f} FPS).")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()