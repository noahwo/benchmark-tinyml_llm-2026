import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5

def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle possible formats like "0: person" or "0 person"
            if line[0].isdigit():
                parts = line.replace(":", " ").split()
                if len(parts) >= 2:
                    labels.append(" ".join(parts[1:]))
                else:
                    labels.append(line)
            else:
                labels.append(line)
    return labels

def get_output_indices(output_details):
    # Try to detect outputs by name where possible, fallback to index order
    idx = {"boxes": None, "classes": None, "scores": None, "num": None}
    for i, od in enumerate(output_details):
        name = od.get("name", "").lower()
        shape = od.get("shape", [])
        if "box" in name or "boxes" in name:
            idx["boxes"] = i
        elif "class" in name or "classes" in name:
            idx["classes"] = i
        elif "score" in name or "scores" in name:
            idx["scores"] = i
        elif "num" in name:
            idx["num"] = i

    # Fallback for typical SSD order if any missing
    if any(v is None for v in idx.values()) and len(output_details) >= 4:
        idx_fallback = {"boxes": 0, "classes": 1, "scores": 2, "num": 3}
        for k, v in idx.items():
            if v is None:
                idx[k] = idx_fallback[k]
    return idx

def preprocess(frame, input_size, input_dtype):
    ih, iw = input_size
    # Resize and convert BGR to RGB
    resized = cv2.resize(frame, (iw, ih))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        tensor = rgb.astype(np.float32) / 255.0
    else:
        tensor = rgb.astype(np.uint8)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

def draw_detections(frame, boxes, classes, scores, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue

        cls_id = int(classes[i])
        # Try direct mapping, else try 1-based
        if 0 <= cls_id < len(labels):
            label_name = labels[cls_id]
        elif 0 <= cls_id + 1 < len(labels):
            label_name = labels[cls_id + 1]
        else:
            label_name = f"id:{cls_id}"

        # Boxes are in normalized [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        # Draw rectangle
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Put label text
        label_text = f"{label_name}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, thickness=-1)
        cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. setup: Initialize interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    out_idx = get_output_indices(output_details)

    # Input size and dtype
    in_det = input_details[0]
    input_shape = in_det["shape"]
    # Expecting [1, height, width, 3]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    input_dtype = in_det["dtype"]

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open input video: {INPUT_PATH}")
        return

    # Prepare writer with input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    frame_count = 0
    t0 = time.time()
    infer_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2. preprocessing
            input_tensor = preprocess(frame, (input_height, input_width), input_dtype)

            # 3. inference
            interpreter.set_tensor(in_det["index"], input_tensor)
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer_end = time.time()
            infer_times.append((t_infer_end - t_infer_start) * 1000.0)  # ms

            # 4. output handling
            boxes = interpreter.get_tensor(output_details[out_idx["boxes"]]["index"])[0]
            classes = interpreter.get_tensor(output_details[out_idx["classes"]]["index"])[0]
            scores = interpreter.get_tensor(output_details[out_idx["scores"]]["index"])[0]
            # num_detections sometimes float; not strictly needed for loop when using scores length

            draw_detections(frame, boxes, classes, scores, labels, CONF_THRESHOLD)

            # Overlay performance info
            avg_ms = np.mean(infer_times[-30:]) if infer_times else 0.0
            elapsed = time.time() - t0
            fps_rt = (frame_count + 1) / elapsed if elapsed > 0 else 0.0
            perf_text = f"Infer: {avg_ms:.1f} ms | FPS: {fps_rt:.1f}"
            cv2.putText(frame, perf_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 10), 2)

            writer.write(frame)
            frame_count += 1

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    avg_infer_ms = np.mean(infer_times) if infer_times else 0.0
    print(f"Processed {frame_count} frames in {total_time:.2f}s | Avg FPS: {avg_fps:.2f} | Avg Inference: {avg_infer_ms:.2f} ms")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()