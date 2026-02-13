import os
import time
import numpy as np
import cv2

# Application: TFLite object detection with TPU
# Target: Google Coral Dev Board
# Note: This script uses a proxy mAP calculation (no ground-truth available). See comments in compute_map_proxy().

# =======================
# Configuration parameters
# =======================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP text overlay
CONFIDENCE_THRESHOLD = 0.5

EDGETPU_SO_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# =======================
# TFLite + EdgeTPU setup
# =======================
# Importing tflite_runtime is required by the app logic for invoking the EdgeTPU-accelerated model.
from tflite_runtime.interpreter import Interpreter, load_delegate


def load_labels(path):
    """
    Load labels from a label map file.
    Supports lines in either:
      - "index label name"
      - "label name"
      - "index: label name"
    Returns a dict id(int) -> name(str)
    """
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
        for i, line in enumerate(raw):
            # Replace ':' with space to support "0: person"
            parts = line.replace(":", " ").split()
            # Try "index label..." format
            idx = None
            if parts and all(ch.isdigit() for ch in parts[0]):
                try:
                    idx = int(parts[0])
                    name = " ".join(parts[1:]).strip() if len(parts) > 1 else str(idx)
                    labels[idx] = name if name else str(idx)
                    continue
                except Exception:
                    idx = None
            # Fallback: use line order as index
            labels[i] = line
    except Exception:
        # If label file missing or unreadable, fallback to empty mapping.
        labels = {}
    return labels


def make_interpreter(model_path, edgetpu_so_path):
    """
    Create and allocate a TFLite Interpreter with EdgeTPU delegate.
    """
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(edgetpu_so_path)]
        )
    except Exception:
        # Fallback: create without delegate if delegate load fails (still runs on CPU but very slow).
        interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def resolve_output_indices(interpreter):
    """
    Resolve indices for detection boxes, classes, scores, and count from output_details.
    Tries name hints; falls back to standard SSD order for TPU models.
    """
    output_details = interpreter.get_output_details()
    name_map = {"boxes": None, "classes": None, "scores": None, "num": None}

    # First pass: try by name hints and shape patterns
    for i, od in enumerate(output_details):
        name = od.get("name", "").lower()
        shape = od.get("shape", [])
        # Heuristics
        if ("box" in name or "bbox" in name) or (len(shape) == 3 and shape[-1] == 4):
            name_map["boxes"] = i if name_map["boxes"] is None else name_map["boxes"]
        elif "class" in name:
            name_map["classes"] = i if name_map["classes"] is None else name_map["classes"]
        elif "score" in name:
            name_map["scores"] = i if name_map["scores"] is None else name_map["scores"]
        elif "num" in name or "count" in name:
            name_map["num"] = i if name_map["num"] is None else name_map["num"]

    # Fallback to common output order: [boxes, classes, scores, num]
    if any(v is None for v in name_map.values()):
        if len(output_details) >= 4:
            name_map = {"boxes": 0, "classes": 1, "scores": 2, "num": 3}
        else:
            # Attempt best-effort from shapes
            # Find boxes = tensor with last dim 4
            for i, od in enumerate(output_details):
                if len(od["shape"]) >= 2 and od["shape"][-1] == 4:
                    name_map["boxes"] = i
            # Remaining assign arbitrarily if still None
            idxs_left = [i for i in range(len(output_details)) if i not in name_map.values() or name_map.values() is None]
            keys_left = [k for k, v in name_map.items() if v is None]
            for k, i in zip(keys_left, idxs_left):
                name_map[k] = i

    return name_map["boxes"], name_map["classes"], name_map["scores"], name_map["num"]


def preprocess_frame(frame_bgr, input_w, input_h, input_dtype):
    """
    Preprocess an OpenCV BGR frame into model input tensor.
    Most EdgeTPU SSD models are quantized uint8 and expect [0..255] RGB input.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.uint8:
        input_data = resized.astype(np.uint8)
    else:
        # Generic float32 fallback: normalize to [0,1]
        input_data = resized.astype(np.float32) / 255.0

    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def postprocess_detections(boxes, classes, scores, count, frame_w, frame_h):
    """
    Convert raw model outputs to a list of detection dicts with absolute pixel boxes.
    boxes are expected normalized [ymin, xmin, ymax, xmax].
    """
    results = []
    n = int(count) if isinstance(count, (int, float, np.integer, np.floating)) else int(count[0] if len(count) else 0)
    n = min(n, len(scores))
    for i in range(n):
        score = float(scores[i])
        cls = int(classes[i])
        y_min, x_min, y_max, x_max = boxes[i]
        # Convert to pixel coordinates and clamp
        y1 = max(0, min(int(y_min * frame_h), frame_h - 1))
        x1 = max(0, min(int(x_min * frame_w), frame_w - 1))
        y2 = max(0, min(int(y_max * frame_h), frame_h - 1))
        x2 = max(0, min(int(x_max * frame_w), frame_w - 1))
        # Ensure top-left is less than bottom-right
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)

        results.append({
            "score": score,
            "class_id": cls,
            "bbox": (x1, y1, x2, y2)  # absolute pixel coords
        })
    return results


def detect_objects(interpreter, io_indices, frame_bgr, input_w, input_h, input_idx, input_dtype):
    """
    Run inference on a single frame and return detections as list of dicts.
    """
    # Preprocess
    input_tensor = preprocess_frame(frame_bgr, input_w, input_h, input_dtype)
    interpreter.set_tensor(input_idx, input_tensor)

    # Inference
    interpreter.invoke()

    # Outputs
    boxes_idx, classes_idx, scores_idx, num_idx = io_indices
    boxes = interpreter.get_tensor(interpreter.get_output_details()[boxes_idx]["index"])[0]
    classes = interpreter.get_tensor(interpreter.get_output_details()[classes_idx]["index"])[0]
    scores = interpreter.get_tensor(interpreter.get_output_details()[scores_idx]["index"])[0]
    num_dets = interpreter.get_tensor(interpreter.get_output_details()[num_idx]["index"])[0]

    # Postprocess
    h, w = frame_bgr.shape[:2]
    dets = postprocess_detections(boxes, classes, scores, num_dets, w, h)
    return dets


def class_color(class_id):
    """
    Deterministic pseudo-color for a class id.
    """
    r = (class_id * 37) % 256
    g = (class_id * 17) % 256
    b = (class_id * 97) % 256
    return int(b), int(g), int(r)  # OpenCV uses BGR


def compute_map_proxy(preds_by_class, pos_count_by_class):
    """
    Compute a proxy mAP without ground-truth:
      - Treat the highest-scoring detection per class per frame as a "proxy" True Positive (TP).
      - All other detections for that class/frame are considered False Positives (FP).
      - For each class: sort all detections by score desc; AP = average of precision@k at each TP.
      - mAP = mean(AP over classes with at least one proxy positive).
    This is a heuristic proxy to produce a stable, reproducible metric for overlay when ground-truth is not available.
    """
    ap_list = []
    for cls_id, total_pos in pos_count_by_class.items():
        if total_pos <= 0:
            continue
        preds = preds_by_class.get(cls_id, [])
        if not preds:
            ap_list.append(0.0)
            continue
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        tp_cum = 0
        fp_cum = 0
        ap_sum = 0.0
        for score, is_tp in preds_sorted:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precision = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0.0
            if is_tp:
                ap_sum += precision
        ap = ap_sum / float(total_pos) if total_pos > 0 else 0.0
        ap_list.append(ap)
    if ap_list:
        return float(sum(ap_list) / len(ap_list))
    return 0.0


def main():
    # Load labels
    labels = load_labels(LABEL_PATH)

    # Setup interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_SO_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]["index"]
    input_h = int(input_details[0]["shape"][1])
    input_w = int(input_details[0]["shape"][2])
    input_dtype = input_details[0]["dtype"]

    # Resolve output indices
    io_indices = resolve_output_indices(interpreter)

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open input video:", INPUT_PATH)
        return

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps is None or in_fps <= 0:
        in_fps = 30.0  # Fallback FPS if metadata missing

    # Prepare output writer
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (in_w, in_h))
    if not writer.isOpened():
        print("ERROR: Could not open output video for writing:", OUTPUT_PATH)
        cap.release()
        return

    # Proxy mAP accumulators
    # preds_by_class: class_id -> list of (score, is_tp)
    preds_by_class = {}
    # pos_count_by_class: class_id -> count of frames where class had at least one detection (proxy positives)
    pos_count_by_class = {}

    # Processing loop
    frame_count = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run detection
        detections = detect_objects(interpreter, io_indices, frame, input_w, input_h, input_idx, input_dtype)

        # Update proxy AP bookkeeping using ALL detections (no score threshold)
        # Determine top detection index per class for this frame
        best_idx_by_class = {}
        for idx, det in enumerate(detections):
            cls = det["class_id"]
            score = det["score"]
            if cls not in best_idx_by_class:
                best_idx_by_class[cls] = idx
            else:
                prev_best_idx = best_idx_by_class[cls]
                if score > detections[prev_best_idx]["score"]:
                    best_idx_by_class[cls] = idx

        # Increment proxy positive count per class (one "GT" per class per frame if present)
        for cls in best_idx_by_class.keys():
            pos_count_by_class[cls] = pos_count_by_class.get(cls, 0) + 1

        # Append predictions per class with TP flag
        for idx, det in enumerate(detections):
            cls = det["class_id"]
            score = float(det["score"])
            is_tp = (best_idx_by_class.get(cls, -1) == idx)
            preds_by_class.setdefault(cls, []).append((score, bool(is_tp)))

        # Compute current proxy mAP
        current_map = compute_map_proxy(preds_by_class, pos_count_by_class)

        # Draw detections above confidence threshold
        for det in detections:
            if det["score"] < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_id"]
            score = det["score"]
            color = class_color(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_name = labels.get(cls, str(cls))
            caption = "{} {:.2f}".format(label_name, score)
            # Text background
            (tw, th), base = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            th2 = th + base + 4
            y_text = max(0, y1 - th2)
            cv2.rectangle(frame, (x1, y_text), (x1 + tw + 6, y_text + th2), color, -1)
            cv2.putText(frame, caption, (x1 + 3, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Overlay mAP (proxy) on frame
        map_text = "mAP (proxy): {:.3f}".format(current_map)
        cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 10), 2, cv2.LINE_AA)

        # Write to output
        writer.write(frame)

    t1 = time.time()
    elapsed = t1 - t0
    total_fps = (frame_count / elapsed) if elapsed > 0 else 0.0
    print("Processed {} frames in {:.2f}s ({:.2f} FPS).".format(frame_count, elapsed, total_fps))
    final_map = compute_map_proxy(preds_by_class, pos_count_by_class)
    print("Final proxy mAP over the video: {:.4f}".format(final_map))
    print("Output saved to:", OUTPUT_PATH)

    # Release resources
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()