import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
output_path = "results/object_detection/test_results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP
confidence_threshold = 0.5

# =========================
# UTILITY FUNCTIONS
# =========================
def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support potential "id label" or just "label" formats
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)
    return labels

def preprocess_frame(frame_bgr, input_w, input_h, input_dtype):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(frame_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    # Normalize/convert dtype
    if input_dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    else:
        inp = resized.astype(np.uint8)
    # Add batch dimension
    return np.expand_dims(inp, axis=0)

def extract_detections(interpreter):
    """Extract detection boxes, classes, scores, and count from model outputs in a robust way."""
    output_details = interpreter.get_output_details()
    boxes = None
    classes = None
    scores = None
    num = None
    for od in output_details:
        arr = interpreter.get_tensor(od['index'])
        # Typical shapes:
        # boxes: [1, num, 4]
        # classes: [1, num]
        # scores: [1, num]
        # num: [1]
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 4:
            boxes = arr[0]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            # Distinguish classes vs scores by value range
            if np.max(arr) <= 1.0 + 1e-6:
                scores = arr[0]
            else:
                classes = arr[0]
        elif arr.ndim == 1 and arr.shape[0] == 1:
            num = int(arr[0])
    # Fallbacks if model omits num (common)
    if boxes is not None:
        n_boxes = boxes.shape[0]
        if num is None:
            num = n_boxes
        else:
            num = min(num, n_boxes)
    # Clip arrays to num
    if boxes is not None:
        boxes = boxes[:num]
    if classes is not None:
        classes = classes[:num]
    if scores is not None:
        scores = scores[:num]
    return boxes, classes, scores, num

def color_for_class(class_id):
    # Deterministic "random-like" color per class id
    np.random.seed(class_id + 12345)
    c = tuple(int(v) for v in np.random.randint(64, 256, size=3))
    return (int(c[0]), int(c[1]), int(c[2]))

def draw_detections(frame_bgr, boxes, classes, scores, labels, threshold, map_value):
    h, w = frame_bgr.shape[:2]
    if boxes is None or classes is None or scores is None:
        return frame_bgr
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]  # normalized [0,1]
        # Convert to absolute coordinates and clamp
        x1 = max(0, min(w - 1, int(x_min * w)))
        y1 = max(0, min(h - 1, int(y_min * h)))
        x2 = max(0, min(w - 1, int(x_max * w)))
        y2 = max(0, min(h - 1, int(y_max * h)))
        cid = int(classes[i]) if not np.isnan(classes[i]) else -1
        label = labels[cid] if (cid >= 0 and cid < len(labels)) else f"id:{cid}"
        color = color_for_class(cid if cid >= 0 else 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx1, ty1 = x1, max(0, y1 - th - 6)
        cv2.rectangle(frame_bgr, (tx1, ty1), (tx1 + tw + 4, ty1 + th + 4), color, -1)
        cv2.putText(frame_bgr, text, (tx1 + 2, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Draw running mAP (proxy) at top-left
    map_text = f"mAP: {map_value:.3f}"
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
    return frame_bgr

def update_map_statistics(per_class_scores, classes, scores, threshold):
    if classes is None or scores is None:
        return
    for i in range(len(scores)):
        s = float(scores[i])
        if s < threshold:
            continue
        cid = int(classes[i]) if not np.isnan(classes[i]) else -1
        if cid < 0:
            continue
        if cid not in per_class_scores:
            per_class_scores[cid] = []
        per_class_scores[cid].append(s)

def compute_map(per_class_scores):
    # Proxy mAP: mean of per-class average detection scores above threshold
    if not per_class_scores:
        return 0.0
    ap_values = []
    for cid, score_list in per_class_scores.items():
        if not score_list:
            continue
        ap_values.append(float(np.mean(score_list)))
    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))

# =========================
# MAIN APPLICATION LOGIC
# =========================
def main():
    # 1. Setup: Load TFLite Interpreter, labels, and input video
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) == 0:
        raise RuntimeError("Model has no input tensors.")

    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']
    input_h = int(input_shape[1])
    input_w = int(input_shape[2])
    input_dtype = input_details[0]['dtype']

    labels = load_labels(label_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    # For mAP (proxy) statistics
    per_class_scores = {}
    running_map = 0.0
    frame_count = 0

    # 2-3-4. Preprocess -> Inference -> Output handling loop
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            input_tensor = preprocess_frame(frame_bgr, input_w, input_h, input_dtype)

            # Inference
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()

            # Extract detections
            boxes, classes, scores, num = extract_detections(interpreter)

            # Update mAP statistics
            update_map_statistics(per_class_scores, classes, scores, confidence_threshold)
            running_map = compute_map(per_class_scores)

            # Draw and write frame
            annotated = draw_detections(frame_bgr.copy(), boxes, classes, scores, labels, confidence_threshold, running_map)
            writer.write(annotated)

    finally:
        cap.release()
        writer.release()

if __name__ == "__main__":
    main()