import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# =========================
# Helper functions
# =========================
def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def get_input_details(interpreter):
    details = interpreter.get_input_details()[0]
    shape = details['shape']
    # Handle dynamic shapes if any (some models may have -1 for batch size)
    if shape[0] <= 0:
        shape[0] = 1
    height, width = int(shape[1]), int(shape[2])
    dtype = details['dtype']
    quant = details.get('quantization', (0.0, 0))
    return details, (height, width), dtype, quant

def preprocess_frame(frame_bgr, input_size, input_dtype, input_quant):
    # Resize and convert color BGR->RGB
    ih, iw = input_size
    resized = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        # Normalize to [0,1]
        input_data = rgb.astype(np.float32) / 255.0
    elif input_dtype == np.uint8:
        scale, zero_point = (0.0, 0)
        if isinstance(input_quant, (list, tuple)) and len(input_quant) == 2:
            scale, zero_point = input_quant
        if scale and scale > 0.0:
            # Quantize from [0,1] float to uint8 using provided scale/zero_point
            input_data = (rgb.astype(np.float32) / 255.0 / scale + zero_point).round().clip(0, 255).astype(np.uint8)
        else:
            # Assume raw 0-255 input
            input_data = rgb.astype(np.uint8)
    else:
        # Fallback: try float32
        input_data = rgb.astype(np.float32) / 255.0

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def resolve_outputs(interpreter):
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(d['index']) for d in output_details]

    boxes = None
    scores = None
    classes = None
    num_detections = None

    # Identify outputs by shapes and value ranges
    for arr in outputs:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr[0]
    # Candidates with shape [1, N]
    candidates_1n = [arr[0] for arr in outputs if arr.ndim == 2 and arr.shape[0] == 1]
    for arr in candidates_1n:
        # Scores typically in [0,1]
        if arr.size > 0 and np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
            scores = arr
        else:
            classes = arr
    # num_detections (optional)
    for arr in outputs:
        if arr.ndim == 1 and arr.size == 1:
            num_detections = int(np.round(arr[0]).astype(int))

    # Fallbacks if some were not identified
    if boxes is None or scores is None or classes is None:
        # Try to map by common order [boxes, classes, scores, num]
        try:
            boxes = outputs[0][0] if boxes is None else boxes
            classes = outputs[1][0] if classes is None else classes
            scores = outputs[2][0] if scores is None else scores
            if len(outputs) > 3 and num_detections is None:
                num_detections = int(np.round(outputs[3][0]).astype(int))
        except Exception:
            pass

    if num_detections is None and scores is not None:
        num_detections = scores.shape[0]

    return boxes, classes, scores, num_detections

def draw_detections(frame, boxes, classes, scores, num_detections, labels, threshold):
    h, w = frame.shape[:2]
    count = 0
    for i in range(min(num_detections, len(scores))):
        score = float(scores[i])
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        # Scale to frame coords
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        cls_id = int(classes[i])
        label = labels[cls_id] if 0 <= cls_id < len(labels) else f"ID {cls_id}"
        caption = f"{label}: {score*100:.1f}%"

        # Color by class id (deterministic)
        color = (37 * (cls_id % 7) + 30, 17 * (cls_id % 13) + 60, 29 * (cls_id % 17) + 90)
        color = tuple(int(c % 255) for c in color)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th_full = th + baseline + 4
        y_text = y1 - 4
        y_bg_top = y_text - th_full + 4
        if y_bg_top < 0:
            y_bg_top = 0
            y_text = th + 2
        cv2.rectangle(frame, (x1, y_bg_top), (x1 + tw + 4, y_bg_top + th_full), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y_bg_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        count += 1
    return count

def put_fps(frame, fps):
    text = f"FPS: {fps:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (10, 10), (10 + tw + 10, 10 + th + baseline + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

# =========================
# 1) SETUP
# =========================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(LABEL_PATH):
        print(f"Error: Label file not found at {LABEL_PATH}")
        return
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input video not found at {INPUT_PATH}")
        return

    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    num_threads = max(1, (os.cpu_count() or 1))
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
    interpreter.allocate_tensors()

    in_detail, in_size, in_dtype, in_quant = get_input_details(interpreter)
    in_height, in_width = in_size

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Input tensor shape: [1, {in_height}, {in_width}, 3], dtype={in_dtype}, quant={in_quant}")
    print(f"Using {num_threads} threads")

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_PATH}")
        return

    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or np.isnan(src_fps) or src_fps <= 0:
        src_fps = 30.0

    ensure_parent_dir(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, src_fps, (src_width, src_height))
    if not writer.isOpened():
        print(f"Error: Could not open writer for {OUTPUT_PATH}")
        cap.release()
        return

    # =========================
    # 2) PREPROCESSING + 3) INFERENCE + 4) OUTPUT HANDLING
    # =========================
    frame_count = 0
    t0 = time.time()
    last_report = t0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = preprocess_frame(frame, (in_height, in_width), in_dtype, in_quant)

            # Set input tensor and run inference
            interpreter.set_tensor(in_detail['index'], input_tensor)
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer_end = time.time()

            # Get outputs
            boxes, classes, scores, num_detections = resolve_outputs(interpreter)
            if boxes is None or classes is None or scores is None or num_detections is None:
                print("Warning: Unable to resolve model outputs; skipping frame.")
                writer.write(frame)
                continue

            # Draw detections
            _ = draw_detections(frame, boxes, classes, scores, num_detections, labels, CONFIDENCE_THRESHOLD)

            # FPS based on inference time (instantaneous)
            infer_time = t_infer_end - t_infer_start
            fps_instant = 1.0 / infer_time if infer_time > 0 else 0.0
            put_fps(frame, fps_instant)

            writer.write(frame)
            frame_count += 1

            # Periodic progress
            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - t0
                overall_fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Processed {frame_count} frames, avg FPS: {overall_fps:.2f}")
                last_report = now

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    print(f"Done. Processed {frame_count} frames in {total_time:.2f}s (avg FPS: {avg_fps:.2f})")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()