import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle common "id label" formats or plain label per line
            # Try to split "index label" if present
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(line)
    return labels

def map_label(labels, class_id):
    # class_id from TFLite SSD models is typically 1-based float (e.g., 1.0 for 'person')
    ci = int(class_id)
    # If label list includes a background/??? at index 0, direct indexing works
    if 0 <= ci < len(labels):
        return labels[ci]
    # Otherwise try 1-based to 0-based shift
    if 0 <= ci - 1 < len(labels):
        return labels[ci - 1]
    return str(ci)

def prepare_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Assume single input
    in_shape = input_details[0]["shape"]
    in_dtype = input_details[0]["dtype"]
    height, width = int(in_shape[1]), int(in_shape[2])
    is_float = (in_dtype == np.float32)
    return interpreter, input_details, output_details, (width, height), is_float

def preprocess_frame(frame_bgr, input_size, is_float_input):
    # Convert BGR->RGB and resize to model input size
    w, h = input_size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)
    if is_float_input:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)
    return input_data

def identify_output_indices(output_details):
    # Try the common SSD order: boxes, classes, scores, num_detections
    if len(output_details) >= 4:
        return 0, 1, 2, 3
    # Fallback heuristic (rarely needed)
    boxes_idx = None
    classes_idx = None
    scores_idx = None
    num_idx = None
    for i, od in enumerate(output_details):
        shape = tuple(od["shape"].tolist())
        if len(shape) == 3 and shape[-1] == 4:
            boxes_idx = i
        elif len(shape) == 1 and shape[0] == 1:
            num_idx = i
    # Identify remaining two as classes and scores (both (1, N))
    remaining = [i for i in range(len(output_details)) if i not in (boxes_idx, num_idx)]
    # Guess order
    if len(remaining) == 2:
        classes_idx, scores_idx = remaining[0], remaining[1]
    return boxes_idx, classes_idx, scores_idx, num_idx

def draw_detections(frame_bgr, boxes, classes, scores, num, labels, threshold):
    h, w = frame_bgr.shape[:2]
    for i in range(int(num)):
        score = float(scores[i])
        if score < threshold:
            continue
        y1, x1, y2, x2 = boxes[i]
        x_min = max(0, min(w - 1, int(x1 * w)))
        y_min = max(0, min(h - 1, int(y1 * h)))
        x_max = max(0, min(w - 1, int(x2 * w)))
        y_max = max(0, min(h - 1, int(y2 * h)))
        cls_id = classes[i]
        label_text = map_label(labels, cls_id)
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, 2)
        caption = f"{label_text}: {score*100:.1f}%"
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y_min, th + 6)
        cv2.rectangle(frame_bgr, (x_min, ty - th - 6), (x_min + tw + 4, ty + baseline - 2), (0, 0, 0), -1)
        cv2.putText(frame_bgr, caption, (x_min + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

    ensure_dir(OUTPUT_PATH)
    labels = load_labels(LABEL_PATH)

    # 1) setup: Initialize interpreter
    interpreter, input_details, output_details, input_size, is_float_input = prepare_interpreter(MODEL_PATH)
    in_index = input_details[0]["index"]
    boxes_idx, classes_idx, scores_idx, num_idx = identify_output_indices(output_details)

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_PATH}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0  # default fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {OUTPUT_PATH}")

    last_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2) preprocessing
            input_tensor = preprocess_frame(frame, input_size, is_float_input)

            # 3) inference
            interpreter.set_tensor(in_index, input_tensor)
            interpreter.invoke()

            # 4) output handling
            outputs = []
            for idx in (boxes_idx, classes_idx, scores_idx, num_idx):
                outputs.append(interpreter.get_tensor(output_details[idx]["index"]))
            boxes = outputs[0][0]
            classes = outputs[1][0]
            scores = outputs[2][0]
            num_detections = int(outputs[3][0])

            draw_detections(frame, boxes, classes, scores, num_detections, labels, CONFIDENCE_THRESHOLD)

            # Optional FPS overlay
            now = time.time()
            dt = max(1e-6, now - last_time)
            last_time = now
            inst_fps = 1.0 / dt
            cv2.putText(frame, f"FPS: {inst_fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

if __name__ == "__main__":
    main()