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
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Support formats:
            #   "0 person" or "person"
            if line[0].isdigit():
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1].strip()
                else:
                    idx = i
                    name = line
            else:
                idx = i
                name = line
            labels[idx] = name
    return labels

def set_input_tensor(interpreter, image_rgb):
    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    input_shape = input_details["shape"]
    input_dtype = input_details["dtype"]

    # Resize to model input size
    height, width = input_shape[1], input_shape[2]
    resized = cv2.resize(image_rgb, (width, height))

    # Add batch dimension
    input_data = np.expand_dims(resized, axis=0)

    # Handle quantization or float models
    if np.issubdtype(input_dtype, np.integer):
        # Quantized input
        scale, zero_point = (1.0, 0)
        if "quantization" in input_details and input_details["quantization"] is not None:
            q = input_details["quantization"]
            if isinstance(q, (list, tuple)) and len(q) == 2:
                scale, zero_point = q
        if scale == 0:
            scale = 1.0
        input_data = input_data.astype(np.float32) / scale + zero_point
        input_data = np.clip(input_data, 0, 255).astype(input_dtype)
    else:
        # Float input: normalize to [0,1]
        input_data = (input_data.astype(np.float32) / 255.0).astype(input_dtype)

    interpreter.set_tensor(input_index, input_data)

def parse_outputs(interpreter):
    output_details = interpreter.get_output_details()
    boxes = classes = scores = num = None

    # First try to use tensor names
    for od in output_details:
        name = str(od.get("name", "")).lower()
        data = interpreter.get_tensor(od["index"])
        if "box" in name:
            boxes = data
        elif "score" in name:
            scores = data
        elif "class" in name:
            classes = data
        elif "num" in name:
            num = data

    # If names didn't resolve everything, use heuristics
    if boxes is None or scores is None or classes is None or num is None:
        candidates = [interpreter.get_tensor(od["index"]) for od in output_details]
        # Identify num: shape (1,) likely
        for c in candidates:
            if isinstance(c, np.ndarray) and c.size == 1:
                num = c
        # Identify boxes: last dim == 4
        for c in candidates:
            if isinstance(c, np.ndarray) and c.ndim >= 2 and c.shape[-1] == 4:
                boxes = c
        # Remaining for scores/classes (shape like (1, N))
        remaining = [c for c in candidates if c is not None and c is not boxes and c is not num]
        # Heuristic: scores are float and within [0,1]
        for c in remaining:
            if c.dtype == np.float32 and np.all((c >= 0) & (c <= 1)):
                scores = c
        for c in remaining:
            if c is not scores:
                classes = c

    # Squeeze batch dim
    if boxes is not None and boxes.ndim > 2:
        boxes = np.squeeze(boxes, axis=0)
    if classes is not None and classes.ndim > 1:
        classes = np.squeeze(classes, axis=0)
    if scores is not None and scores.ndim > 1:
        scores = np.squeeze(scores, axis=0)
    if num is not None:
        num = int(np.squeeze(num).astype(np.int32))

    return boxes, classes, scores, num

def draw_detections(frame_bgr, boxes, classes, scores, num, labels, threshold):
    h, w = frame_bgr.shape[:2]
    drawn = 0
    for i in range(num):
        score = float(scores[i]) if scores is not None else 0.0
        if score < threshold:
            continue
        cls_id = int(classes[i]) if classes is not None else -1
        label = labels.get(cls_id, f"#{cls_id}")
        # Boxes are [ymin, xmin, ymax, xmax] normalized
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = int(max(0, xmin * w))
        y1 = int(max(0, ymin * h))
        x2 = int(min(w - 1, xmax * w))
        y2 = int(min(h - 1, ymax * h))

        # Color derived from class id
        color = ((37 * (cls_id + 1)) % 255, (17 * (cls_id + 1)) % 255, (29 * (cls_id + 1)) % 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        caption = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame_bgr, caption, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        drawn += 1
    return drawn

# =========================
# 1. Setup
# =========================
def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input size from model
    input_details = interpreter.get_input_details()[0]
    in_h, in_w = int(input_details["shape"][1]), int(input_details["shape"][2])

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open input video: {input_path}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or np.isnan(src_fps) or src_fps <= 0:
        src_fps = 30.0

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, src_fps, (src_w, src_h))
    if not out.isOpened():
        print(f"ERROR: Cannot open output video for writing: {output_path}")
        cap.release()
        return

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            # =========================
            # 2. Preprocessing
            # =========================
            # Convert BGR (OpenCV) to RGB (model)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            set_input_tensor(interpreter, frame_rgb)

            # =========================
            # 3. Inference
            # =========================
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer = time.time() - t_infer_start

            # Retrieve outputs
            boxes, classes, scores, num = parse_outputs(interpreter)
            if num is None:
                num = len(scores) if scores is not None else 0

            # =========================
            # 4. Output Handling
            # =========================
            drawn = 0
            if boxes is not None and classes is not None and scores is not None:
                drawn = draw_detections(frame_bgr, boxes, classes, scores, num, labels, confidence_threshold)

            # Overlay FPS and info
            total_fps = 1.0 / max(t_infer, 1e-6)
            cv2.putText(frame_bgr, f"Detections: {drawn}  Inference FPS: {total_fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

            out.write(frame_bgr)

    finally:
        cap.release()
        out.release()

    elapsed = time.time() - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Processing complete.")
    print(f"Frames processed: {frame_count}")
    print(f"Average throughput: {avg_fps:.2f} FPS")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()