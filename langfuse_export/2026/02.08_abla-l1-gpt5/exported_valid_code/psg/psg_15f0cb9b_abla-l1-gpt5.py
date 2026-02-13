import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(label_path):
    labels = []
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Handle simple label files; ignore proto-style labelmaps
                if line.startswith("item {") or line.startswith("id:") or line.startswith("name:") or line.startswith("display_name:") or line.startswith("}"):
                    # Skip proto-style lines (not supported here)
                    continue
                labels.append(line)
    return labels

def get_label(labels, class_id):
    if not labels:
        return str(class_id)
    cid = int(class_id)
    # Try direct index
    if 0 <= cid < len(labels):
        return labels[cid]
    # Try 1-based indexing common in TFLite detection models
    if 1 <= cid <= len(labels):
        return labels[cid - 1]
    return str(class_id)

def prepare_input_tensor(frame_bgr, input_details):
    # Input tensor info
    input_shape = input_details[0]["shape"]  # e.g., [1, height, width, 3]
    height, width = int(input_shape[1]), int(input_shape[2])
    dtype = input_details[0]["dtype"]

    # Preprocessing: resize + BGR->RGB
    resized = cv2.resize(frame_bgr, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if dtype == np.float32:
        tensor = (rgb.astype(np.float32) / 255.0).reshape(1, height, width, 3)
    else:
        # For quantized uint8 models, feed raw 0-255
        tensor = rgb.astype(np.uint8).reshape(1, height, width, 3)
    return tensor

def draw_detections(frame_bgr, boxes, classes, scores, labels, score_threshold):
    h, w = frame_bgr.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < score_threshold:
            continue
        y1, x1, y2, x2 = boxes[i]
        # Boxes are normalized [0,1]
        x_min = int(max(0, min(w, x1 * w)))
        y_min = int(max(0, min(h, y1 * h)))
        x_max = int(max(0, min(w, x2 * w)))
        y_max = int(max(0, min(h, y2 * h)))

        class_id = int(classes[i])
        label_text = get_label(labels, class_id)
        caption = f"{label_text}: {score:.2f}"

        # Draw rectangle
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        rect_top = max(0, y_min - th - baseline - 4)
        rect_bottom = y_min
        cv2.rectangle(frame_bgr, (x_min, rect_top), (x_min + tw + 6, rect_bottom), color, -1)
        cv2.putText(frame_bgr, caption, (x_min + 3, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame_bgr

def main():
    # 1. Setup
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    labels = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {INPUT_PATH}")
        return

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or np.isnan(in_fps) or in_fps <= 0:
        in_fps = 30.0  # fallback

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (orig_w, orig_h))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    frame_count = 0
    t0 = time.time()

    # 2-3-4. Preprocessing -> Inference -> Output handling (per frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Preprocessing
        input_tensor = prepare_input_tensor(frame, input_details)
        interpreter.set_tensor(input_details[0]["index"], input_tensor)

        # 3. Inference
        interpreter.invoke()

        # 4. Output handling
        # Typical TFLite SSD outputs: boxes [1, num, 4], classes [1, num], scores [1, num], num_detections [1]
        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        # num = int(interpreter.get_tensor(output_details[3]["index"])[0])  # not strictly needed

        annotated = draw_detections(frame, boxes, classes, scores, labels, CONFIDENCE_THRESHOLD)
        writer.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - t0
            fps_proc = frame_count / max(elapsed, 1e-6)
            print(f"Processed {frame_count} frames - approx {fps_proc:.2f} FPS")

    cap.release()
    writer.release()
    total_time = time.time() - t0
    print(f"Completed. Processed {frame_count} frames in {total_time:.2f}s ({(frame_count/max(total_time,1e-6)):.2f} FPS).")
    print(f"Saved output to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()