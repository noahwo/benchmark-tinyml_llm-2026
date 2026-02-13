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


# =========================
# Utility Functions
# =========================
def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def preprocess_frame(frame_bgr, input_size, input_dtype):
    # Convert BGR to RGB, resize to model's expected input size, and normalize if needed
    ih, iw = input_size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def parse_detection_outputs(interpreter, output_details):
    # Retrieve all output tensors, then assign by shape/value characteristics
    tensors = [interpreter.get_tensor(od["index"]) for od in output_details]
    boxes = classes = scores = num = None

    for t in tensors:
        shape = t.shape
        if len(shape) == 3 and shape[-1] == 4:
            boxes = t
        elif len(shape) == 2:
            # Distinguish between classes and scores by value range
            if t.dtype == np.float32 and np.max(t) <= 1.0:
                scores = t
            else:
                classes = t
        elif len(shape) == 1 and shape[0] == 1:
            # num_detections
            num = int(t[0])

    # Fallbacks if not provided
    if boxes is None or classes is None or scores is None:
        raise RuntimeError("Unexpected model outputs. Could not find boxes/classes/scores tensors.")

    # Flatten batch dimension
    boxes = boxes[0]
    classes = classes[0].astype(np.int32)
    scores = scores[0].astype(np.float32)
    if num is None:
        num = len(scores)

    return boxes, classes, scores, num


def draw_detections(frame_bgr, boxes, classes, scores, num, labels, conf_threshold):
    h, w = frame_bgr.shape[:2]
    for i in range(num):
        score = float(scores[i])
        if score < conf_threshold:
            continue

        # boxes are in normalized ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        class_id = int(classes[i])
        label_text = labels[class_id] if 0 <= class_id < len(labels) else f"id {class_id}"
        caption = f"{label_text}: {score:.2f}"

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(frame_bgr, (x1, ty1), (x1 + tw + 6, ty1 + th + baseline + 6), color, thickness=-1)
        cv2.putText(frame_bgr, caption, (x1 + 3, ty1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    # 1. Setup
    labels = load_labels(label_path)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Model input info
    in_shape = input_details[0]["shape"]  # [1, H, W, 3]
    in_h, in_w = int(in_shape[1]), int(in_shape[2])
    in_dtype = input_details[0]["dtype"]

    # Video I/O
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir_for_file(output_path)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # Processing loop
    frame_count = 0
    t_start_total = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Preprocessing
        input_data = preprocess_frame(frame, (in_h, in_w), in_dtype)

        # 3. Inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        t0 = time.time()
        interpreter.invoke()
        inf_ms = (time.time() - t0) * 1000.0

        boxes, classes, scores, num = parse_detection_outputs(interpreter, output_details)

        # 4. Output handling (draw and write)
        draw_detections(frame, boxes, classes, scores, num, labels, confidence_threshold)
        # Overlay inference time
        inf_text = f"Inference: {inf_ms:.1f} ms"
        cv2.putText(frame, inf_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, inf_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        writer.write(frame)
        frame_count += 1

    # Cleanup
    cap.release()
    writer.release()

    total_time = time.time() - t_start_total
    if total_time > 0 and frame_count > 0:
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f}s (avg {avg_fps:.2f} FPS).")
    else:
        print("No frames processed.")


if __name__ == "__main__":
    main()