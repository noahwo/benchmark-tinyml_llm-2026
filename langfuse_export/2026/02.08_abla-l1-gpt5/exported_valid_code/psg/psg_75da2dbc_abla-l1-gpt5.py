import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# ==============================
# Configuration Parameters
# ==============================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# ==============================
# Utilities
# ==============================
def load_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    label_offset = 1 if labels and labels[0] == "???" else 0
    return labels, label_offset

def preprocess_frame(frame, input_size, expects_float):
    # Convert BGR (OpenCV) to RGB (model)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size[1], input_size[0]))  # (width, height)
    if expects_float:
        input_tensor = resized.astype(np.float32) / 255.0
    else:
        input_tensor = resized.astype(np.uint8)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def draw_detections(frame, boxes, classes, scores, num, labels, label_offset, threshold):
    h, w = frame.shape[:2]
    color = (0, 255, 0)
    thickness = max(1, int(round(0.002 * (h + w))))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.8, (h + w) / 2000.0))
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        cls_id = int(classes[i])
        label_idx = cls_id + label_offset
        label_text = labels[label_idx] if 0 <= label_idx < len(labels) else str(cls_id)

        y_min, x_min, y_max, x_max = boxes[i]
        x1 = int(max(0, min(w - 1, x_min * w)))
        y1 = int(max(0, min(h - 1, y_min * h)))
        x2 = int(max(0, min(w - 1, x_max * w)))
        y2 = int(max(0, min(h - 1, y_max * h)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text = f"{label_text}: {score*100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Background for text
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 3)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return frame

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ==============================
# Main pipeline
# ==============================
def main():
    # 1. Setup
    labels, label_offset = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input tensor info
    in_detail = input_details[0]
    input_index = in_detail["index"]
    # Expected shape: [1, height, width, 3]
    in_shape = in_detail["shape"]
    in_height, in_width = int(in_shape[1]), int(in_shape[2])
    expects_float = (in_detail["dtype"] == np.float32)

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {INPUT_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ensure_dir(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    print("Model loaded.")
    print(f"Input tensor size: {in_width}x{in_height}, dtype: {'float32' if expects_float else 'uint8'}")
    print(f"Processing video: {INPUT_PATH}")
    print(f"Saving results to: {OUTPUT_PATH}")

    frame_count = 0
    t0 = time.time()

    # 2-3-4. Preprocessing -> Inference -> Output handling
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # Preprocessing
        input_tensor = preprocess_frame(frame, (in_height, in_width), expects_float)

        # Inference
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

        # Output handling
        # Typical SSD MobileNet v1 TFLite output order:
        # 0: boxes [1, num, 4], 1: classes [1, num], 2: scores [1, num], 3: num_detections [1]
        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        num = int(interpreter.get_tensor(output_details[3]["index"])[0])

        frame = draw_detections(frame, boxes, classes, scores, num, labels, label_offset, CONFIDENCE_THRESHOLD)

        # Overlay inference FPS
        dt = time.time() - start
        inf_fps = 1.0 / dt if dt > 0 else 0.0
        info_text = f"FPS: {inf_fps:.2f}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    total_time = time.time() - t0
    if total_time > 0:
        print(f"Done. Processed {frame_count} frames in {total_time:.2f}s ({frame_count/total_time:.2f} FPS avg).")
    else:
        print(f"Done. Processed {frame_count} frames.")

if __name__ == "__main__":
    main()