import os
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# ==============================
# Configuration (provided)
# ==============================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# ==============================
# Utility functions
# ==============================
def load_labels(path):
    """
    Loads label map as {int_id: label}.
    Supports formats:
      - "id label" per line
      - "id: label" per line
      - or one label per line (index based)
    """
    labels = {}
    if not os.path.exists(path):
        print(f"Warning: Label file not found at {path}. Using numeric class ids.")
        return labels
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for idx, line in enumerate(lines):
        if ":" in line:
            parts = line.split(":", 1)
        else:
            parts = line.split(maxsplit=1)
        try:
            if len(parts) == 2 and parts[0].strip().isdigit():
                labels[int(parts[0].strip())] = parts[1].strip()
            else:
                labels[idx] = line
        except Exception:
            labels[idx] = line
    return labels

def get_label_name(labels, class_id):
    """Resolve label name from dict; handle 0- or 1-based label ids."""
    if class_id in labels:
        return labels[class_id]
    if (class_id + 1) in labels:
        return labels[class_id + 1]
    return f"id_{class_id}"

def class_color(class_id):
    """Deterministic distinct BGR color per class id."""
    # Simple hashing to generate a color
    np.random.seed(class_id + 12345)
    color = tuple(int(x) for x in np.random.choice(range(64, 256), size=3))
    # OpenCV uses BGR
    return (color[2], color[1], color[0])

def preprocess_frame(frame_bgr, input_shape, input_details):
    """
    Preprocess frame:
    - Convert BGR to RGB
    - Resize to model input size
    - Quantize or convert dtype as required by model
    Returns input tensor of shape [1, H, W, 3]
    """
    _, in_h, in_w, _ = input_shape
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    dtype = input_details[0]["dtype"]
    quant_params = input_details[0].get("quantization_parameters", {})
    scales = quant_params.get("scales", [])
    zero_points = quant_params.get("zero_points", [])
    scale = scales[0] if len(scales) > 0 else 0.0
    zero_point = zero_points[0] if len(zero_points) > 0 else 0

    if dtype == np.uint8:
        if scale and scale > 0:
            input_data = np.clip(np.round(resized / scale + zero_point), 0, 255).astype(np.uint8)
        else:
            input_data = resized.astype(np.uint8)
    else:
        # Default float input normalization to [0,1]
        input_data = resized.astype(np.float32) / 255.0

    return np.expand_dims(input_data, axis=0)

def resolve_output_indices(output_details):
    """
    Resolve indices for boxes, classes, scores, count using output tensor names and shapes.
    Falls back to common SSD order if names are unavailable.
    """
    boxes_idx = classes_idx = scores_idx = count_idx = None
    for i, od in enumerate(output_details):
        name = od.get("name", "").lower()
        shape = od.get("shape", [])
        if "box" in name:
            boxes_idx = i
        elif "class" in name:
            classes_idx = i
        elif "score" in name:
            scores_idx = i
        elif "count" in name or "num" in name:
            count_idx = i

    # Fallback if names did not help
    if boxes_idx is None or classes_idx is None or scores_idx is None:
        # Typical TFLite SSD output order: [boxes, classes, scores, count]
        if len(output_details) >= 3:
            boxes_idx = 0 if boxes_idx is None else boxes_idx
            classes_idx = 1 if classes_idx is None else classes_idx
            scores_idx = 2 if scores_idx is None else scores_idx
        if len(output_details) >= 4 and count_idx is None:
            count_idx = 3

    return boxes_idx, classes_idx, scores_idx, count_idx

def compute_proxy_map(all_scores):
    """
    Compute a proxy mAP using only detection confidences (no ground truth available):
    For thresholds t in [0.50, 0.55, ..., 0.95], precision(t) = (#scores >= t) / (total #scores).
    proxy_mAP = average over thresholds.
    """
    if len(all_scores) == 0:
        return 0.0
    arr = np.array(all_scores, dtype=np.float32)
    thresholds = [0.50 + i * 0.05 for i in range(10)]  # 0.50..0.95
    precisions = [(arr >= t).sum() / float(len(arr)) for t in thresholds]
    return float(np.mean(precisions))

def draw_detections(frame_bgr, detections, labels, map_value):
    """
    Draw bounding boxes, labels, and the running proxy mAP on the frame.
    detections: list of (xmin, ymin, xmax, ymax, class_id, score)
    """
    for (xmin, ymin, xmax, ymax, cls_id, score) in detections:
        color = class_color(cls_id)
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, 2)
        label_text = f"{get_label_name(labels, cls_id)} {score:.2f}"
        # Background for text
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (xmin, max(0, ymin - th - baseline)), (xmin + tw + 2, ymin), color, -1)
        cv2.putText(frame_bgr, label_text, (xmin + 1, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw proxy mAP at top-left corner
    map_text = f"mAP: {map_value:.3f}"
    (tw, th), baseline = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (5, 5), (5 + tw + 8, 5 + th + baseline + 8), (0, 0, 0), -1)
    cv2.putText(frame_bgr, map_text, (9, 9 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ==============================
# Main application
# ==============================
def main():
    # 1. Setup: interpreter with EdgeTPU, labels, and video IO
    print("Loading labels...")
    labels = load_labels(LABEL_PATH)

    print("Initializing TFLite interpreter with EdgeTPU delegate...")
    try:
        interpreter = Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0")]
        )
    except ValueError as e:
        print("Failed to load EdgeTPU delegate. Ensure the library path is correct and model is compiled for EdgeTPU.")
        raise e

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]

    boxes_idx, classes_idx, scores_idx, count_idx = resolve_output_indices(output_details)
    if boxes_idx is None or classes_idx is None or scores_idx is None:
        raise RuntimeError("Could not resolve output tensor indices for boxes/classes/scores.")

    print(f"Model input shape: {input_shape}")

    print(f"Opening input video: {INPUT_PATH}")
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video at: {INPUT_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps is None or in_fps <= 0:
        in_fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (frame_w, frame_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video for writing at: {OUTPUT_PATH}")

    print("Processing video...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    frame_idx = 0
    all_scores = []  # for proxy mAP

    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 2. Preprocess
        input_tensor = preprocess_frame(frame, input_shape, input_details)

        # 3. Inference
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        t_infer_start = time.time()
        interpreter.invoke()
        t_infer = (time.time() - t_infer_start) * 1000.0

        # 4. Output handling: parse detections
        boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]["index"])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]["index"])[0]
        if count_idx is not None:
            num = int(np.squeeze(interpreter.get_tensor(output_details[count_idx]["index"])))
        else:
            num = len(scores)

        # Accumulate scores for proxy mAP (use only valid numeric scores > 0)
        valid_scores = [float(s) for s in scores[:num] if np.isfinite(s) and s > 0.0]
        all_scores.extend(valid_scores)

        detections = []
        for i in range(num):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(classes[i])
            # boxes are [ymin, xmin, ymax, xmax] normalized to [0,1]
            ymin = max(0, min(1, float(boxes[i][0])))
            xmin = max(0, min(1, float(boxes[i][1])))
            ymax = max(0, min(1, float(boxes[i][2])))
            xmax = max(0, min(1, float(boxes[i][3])))

            x0 = int(xmin * frame_w)
            y0 = int(ymin * frame_h)
            x1 = int(xmax * frame_w)
            y1 = int(ymax * frame_h)
            # clamp
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(frame_w - 1, x1), min(frame_h - 1, y1)
            detections.append((x0, y0, x1, y1, cls_id, score))

        # Compute running proxy mAP and draw overlays
        proxy_map = compute_proxy_map(all_scores)
        draw_detections(frame, detections, labels, proxy_map)

        out.write(frame)

        # Optional: console progress
        if frame_idx % 30 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            if total_frames > 0:
                print(f"[{frame_idx}/{total_frames}] FPS: {fps:.2f} | last infer: {t_infer:.1f} ms | detections: {len(detections)}")
            else:
                print(f"[{frame_idx}] FPS: {fps:.2f} | last infer: {t_infer:.1f} ms | detections: {len(detections)}")

    cap.release()
    out.release()

    final_map = compute_proxy_map(all_scores)
    print(f"Processing complete. Output saved to: {OUTPUT_PATH}")
    print(f"Final proxy mAP over video: {final_map:.4f}")

if __name__ == "__main__":
    main()