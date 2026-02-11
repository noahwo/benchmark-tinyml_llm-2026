import os
import time
import numpy as np
import cv2

# Per guideline: use the AI Edge LiteRT interpreter
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Output processed video
CONFIDENCE_THRESHOLD = 0.5  # Threshold for displaying detections


def load_labels(label_path):
    """
    Load labels from the provided label map.
    Supports "id label" or "label" per line formats.
    Returns a dict mapping class_id (int) -> label (str).
    """
    labels = {}
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try "id label" format first
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                class_id = int(parts[0])
                label = parts[1].strip()
                labels[class_id] = label
            else:
                # Fallback: assume simple "label" with implicit index
                labels[idx] = line
    return labels


def ensure_dir_for_file(file_path):
    """Ensure the parent directory exists for the given file path."""
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def preprocess_frame(frame_bgr, input_shape, input_dtype):
    """
    Preprocess a BGR frame to match the model input requirements.
    - Resize to model's expected (H, W)
    - Convert BGR to RGB
    - Normalize if float input
    - Add batch dimension
    Returns the prepared input tensor.
    """
    _, in_h, in_w, in_c = input_shape  # Expect shape [1, H, W, C]
    resized = cv2.resize(frame_bgr, (in_w, in_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        input_data = rgb.astype(np.float32) / 255.0
    else:
        # Assume uint8 quantized
        input_data = rgb.astype(np.uint8)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def identify_detection_tensors(interpreter):
    """
    Identify and fetch detection outputs from the interpreter in a robust way.
    Returns a dict with keys: 'boxes', 'classes', 'scores', 'count'
      - boxes: (N, 4) array with [ymin, xmin, ymax, xmax] normalized to [0,1] (typical SSD output)
      - classes: (N,) array (float or int). If unavailable, returns None.
      - scores: (N,) float array. If unavailable, returns None.
      - count: int number of detections (if provided), else deduced from shapes
    This function avoids assumptions about output tensor order and guards against None values.
    """
    outputs = interpreter.get_output_details()
    arrays = []
    meta = []
    for od in outputs:
        name = (od.get("name") or "").lower()
        tensor = interpreter.get_tensor(od["index"])
        arr = np.squeeze(tensor)  # remove leading dims like [1, ...]
        arrays.append(arr)
        meta.append({"name": name, "shape": arr.shape, "dtype": arr.dtype})

    boxes = None
    classes = None
    scores = None
    count = None

    # Pass 1: Identify by name hints
    for arr, m in zip(arrays, meta):
        name = m["name"]
        shape = m["shape"]
        if isinstance(shape, tuple) and len(shape) == 2 and shape[1] == 4 and ("box" in name or "bbox" in name or "loc" in name or ":0" in name):
            boxes = arr
        elif len(np.atleast_1d(arr).shape) == 1 and ("class" in name or "label" in name or ":1" in name):
            classes = arr
        elif len(np.atleast_1d(arr).shape) == 1 and ("score" in name or "confidence" in name or "prob" in name or ":2" in name):
            scores = arr
        elif (np.issubdtype(m["dtype"], np.integer) and (("num" in name) or ("count" in name) or name.endswith(":3"))):
            try:
                count = int(np.array(arr).flatten()[0])
            except Exception:
                pass

    # Pass 2: Fallback by shape if still missing
    if boxes is None:
        for arr, m in zip(arrays, meta):
            shape = m["shape"]
            if isinstance(shape, tuple) and len(shape) == 2 and shape[1] == 4:
                boxes = arr
                break

    # Determine N (number of candidates)
    N = 0
    if boxes is not None and len(boxes.shape) == 2 and boxes.shape[1] == 4:
        N = boxes.shape[0]
    elif scores is not None and np.ndim(scores) == 1:
        N = scores.shape[0]
    elif classes is not None and np.ndim(classes) == 1:
        N = classes.shape[0]

    if count is None:
        # Some models don't provide a separate count tensor
        count = int(N)

    # If classes or scores not identified by name, try to infer by shape relative to N
    if classes is None:
        for arr, m in zip(arrays, meta):
            if np.ndim(arr) == 1 and arr.shape[0] == N and (np.issubdtype(m["dtype"], np.integer) or np.issubdtype(m["dtype"], np.floating)):
                # Avoid picking scores if already set
                if scores is not None and np.allclose(arr, scores):
                    continue
                classes = arr
                break

    if scores is None:
        # Pick a 1D float array of length N as scores
        for arr, m in zip(arrays, meta):
            if np.ndim(arr) == 1 and arr.shape[0] == N and np.issubdtype(m["dtype"], np.floating):
                if classes is not None and np.allclose(arr, classes):
                    continue
                scores = arr
                break

    # Normalize dtypes and shapes
    if boxes is not None:
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    if classes is not None:
        classes = np.array(classes)
    if scores is not None:
        scores = np.array(scores, dtype=np.float32).reshape(-1)

    # Safety cap: ensure count does not exceed available lengths
    if boxes is not None:
        count = min(count, boxes.shape[0])
    if classes is not None:
        count = min(count, classes.shape[0])
    if scores is not None:
        count = min(count, scores.shape[0])

    return {"boxes": boxes, "classes": classes, "scores": scores, "count": int(count)}


def build_detections(output_dict, frame_w, frame_h, confidence_threshold, labels):
    """
    Build a list of detection dicts from model outputs.
    Each detection dict contains:
      - 'box': (x1, y1, x2, y2) in pixel coordinates
      - 'score': float confidence
      - 'class_id': int id (or -1 if unknown)
      - 'label': string label
    This function avoids the previous crash by guarding None outputs.
    """
    boxes = output_dict.get("boxes", None)
    classes = output_dict.get("classes", None)
    scores = output_dict.get("scores", None)
    count = int(output_dict.get("count", 0))

    detections = []

    if scores is None or boxes is None or count <= 0:
        return detections  # Nothing to show

    # Ensure shapes
    count = min(count, boxes.shape[0], scores.shape[0])

    for i in range(count):
        score = float(scores[i])
        if score < confidence_threshold:
            continue

        # SSD boxes are typically [ymin, xmin, ymax, xmax] normalized to [0,1]
        ymin, xmin, ymax, xmax = boxes[i].tolist()

        # Convert to pixel coordinates
        x1 = int(max(0, xmin) * frame_w)
        y1 = int(max(0, ymin) * frame_h)
        x2 = int(min(1.0, xmax) * frame_w)
        y2 = int(min(1.0, ymax) * frame_h)

        # Guard for classes being None
        if classes is not None and i < classes.shape[0]:
            try:
                class_id = int(classes[i])
            except Exception:
                # Some models provide float class IDs
                class_id = int(float(classes[i]))
        else:
            class_id = -1

        # Label resolution
        if class_id in (labels or {}):
            label_str = labels[class_id]
        elif class_id >= 0:
            label_str = f"id_{class_id}"
        else:
            label_str = "N/A"

        detections.append({
            "box": (x1, y1, x2, y2),
            "score": score,
            "class_id": class_id,
            "label": label_str
        })

    return detections


def draw_detections(frame, detections, map_value=None, fps=None):
    """
    Draw rectangles and label texts for each detection.
    Optionally overlay mAP and FPS texts.
    """
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        score = det["score"]
        color = (0, 255, 0)  # Green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - 10)
        cv2.rectangle(frame, (x1, y_text - th - baseline), (x1 + tw, y_text + baseline // 2), color, -1)
        cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay metrics
    overlay_y = 20
    if map_value is not None:
        cv2.putText(frame, f"mAP (proxy): {map_value:.3f}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)
        overlay_y += 22
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)


def compute_proxy_map(per_class_scores):
    """
    Compute a proxy mAP from accumulated detection confidences per class.
    Note: True mAP requires ground truth. Since only a video input is provided (no annotations),
    we compute a simple proxy: average of mean confidences across classes that had detections.
    """
    class_means = []
    for cls_id, scores in per_class_scores.items():
        if scores:  # non-empty
            class_means.append(float(np.mean(scores)))
    if not class_means:
        return 0.0
    return float(np.mean(class_means))


def main():
    # 1) Setup: Load interpreter, allocate tensors, load labels, open input video
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter
    num_threads = max(1, min(4, os.cpu_count() or 1))
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    if not input_details:
        raise RuntimeError("Interpreter has no input details.")
    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # sensible default for Raspberry Pi

    # Prepare output writer
    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not out_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output writer: {OUTPUT_PATH}")

    # 2-4) Loop: preprocess, inference, postprocess, draw, compute proxy mAP, save
    per_class_scores = {}  # class_id -> list of confidences
    prev_time = time.time()
    avg_fps = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_data = preprocess_frame(frame, input_shape, input_dtype)

        # Inference
        interpreter.set_tensor(input_index, input_data)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()

        # Extract detections robustly (fixes prior NoneType issue)
        output_dict = identify_detection_tensors(interpreter)
        detections = build_detections(output_dict, frame_w, frame_h, CONFIDENCE_THRESHOLD, labels)

        # Accumulate scores for proxy mAP
        for det in detections:
            cls = det["class_id"]
            if cls not in per_class_scores:
                per_class_scores[cls] = []
            per_class_scores[cls].append(det["score"])
        proxy_map = compute_proxy_map(per_class_scores)

        # FPS smoothing
        inst_fps = 1.0 / max(1e-6, (t1 - t0))
        if avg_fps is None:
            avg_fps = inst_fps
        else:
            avg_fps = 0.9 * avg_fps + 0.1 * inst_fps

        # Draw and write
        draw_detections(frame, detections, map_value=proxy_map, fps=avg_fps)
        out_writer.write(frame)

    # Cleanup
    cap.release()
    out_writer.release()

    # Final log
    print(f"Processing complete.")
    print(f"Input video:  {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()