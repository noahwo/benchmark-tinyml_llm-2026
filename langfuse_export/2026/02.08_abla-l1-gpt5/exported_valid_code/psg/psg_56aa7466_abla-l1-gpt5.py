import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def prepare_input(frame_bgr, input_details):
    # Get expected input size and dtype
    _, in_h, in_w, _ = input_details[0]["shape"]
    in_dtype = input_details[0]["dtype"]

    # Preprocessing: BGR -> RGB, resize, expand dims, dtype conversion
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)

    if in_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(in_dtype)
    return input_data

def map_output_indices(output_details):
    boxes_idx = scores_idx = classes_idx = num_idx = None
    # Try to infer from output tensor names
    for i, od in enumerate(output_details):
        name = od.get("name", "").lower()
        if "box" in name:
            boxes_idx = i
        elif "score" in name:
            scores_idx = i
        elif "class" in name:
            classes_idx = i
        elif "num" in name:
            num_idx = i

    # Fallback to common SSD ordering if names are unavailable
    if boxes_idx is None or scores_idx is None or classes_idx is None:
        if len(output_details) >= 3:
            boxes_idx = 0 if boxes_idx is None else boxes_idx
            classes_idx = 1 if classes_idx is None else classes_idx
            scores_idx = 2 if scores_idx is None else scores_idx
        if len(output_details) >= 4 and num_idx is None:
            num_idx = 3
    return boxes_idx, scores_idx, classes_idx, num_idx

def draw_detections(frame_bgr, boxes, classes, scores, num, labels, threshold):
    h, w = frame_bgr.shape[:2]
    label_offset = 1  # common for SSD MobileNet label maps where label 0 is "???"
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        cls_id = int(classes[i])
        label_index = cls_id + label_offset
        label_text = labels[label_index] if 0 <= label_index < len(labels) else str(cls_id)

        ymin, xmin, ymax, xmax = boxes[i]
        left = int(xmin * w)
        top = int(ymin * h)
        right = int(xmax * w)
        bottom = int(ymax * h)

        left = max(0, min(left, w - 1))
        right = max(0, min(right, w - 1))
        top = max(0, min(top, h - 1))
        bottom = max(0, min(bottom, h - 1))

        # Draw bounding box
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label and score
        caption = f"{label_text}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (left, top - th - baseline), (left + tw, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame_bgr, caption, (left, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    # 1. Setup
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label map not found at {label_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found at {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labels = load_labels(label_path)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    boxes_idx, scores_idx, classes_idx, num_idx = map_output_indices(output_details)

    # Video I/O setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 1e-2:
        input_fps = 30.0  # fallback if FPS is not available

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 2. Preprocessing
            input_data = prepare_input(frame, input_details)

            # 3. Inference
            interpreter.set_tensor(input_details[0]["index"], input_data)
            t_infer_start = time.time()
            interpreter.invoke()
            infer_time = (time.time() - t_infer_start) * 1000.0  # ms

            # 4. Output handling
            boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[0]
            scores = interpreter.get_tensor(output_details[scores_idx]["index"])[0]
            classes = interpreter.get_tensor(output_details[classes_idx]["index"])[0]
            if num_idx is not None:
                num = int(np.squeeze(interpreter.get_tensor(output_details[num_idx]["index"])))
            else:
                num = len(scores)

            draw_detections(frame, boxes, classes, scores, num, labels, confidence_threshold)

            # Optional: annotate inference time
            text = f"Inference: {infer_time:.1f} ms"
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 10), 2)

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    if frame_count > 0 and total_time > 0:
        print(f"Processed {frame_count} frames in {total_time:.2f}s ({frame_count/total_time:.2f} FPS).")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()