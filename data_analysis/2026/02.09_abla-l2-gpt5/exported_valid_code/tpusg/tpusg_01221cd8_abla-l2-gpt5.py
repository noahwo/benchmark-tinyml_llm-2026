import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"  # Save processed video with boxes, labels, and mAP
CONFIDENCE_THRESHOLD = 0.5

EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


# =========================
# Utility: Labels
# =========================
def load_labels(path):
    # Attempts to parse both simple label files and pbtxt-style label maps.
    labels = {}
    if not os.path.exists(path):
        return labels

    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Try pbtxt-style
    if any("item" in ln for ln in lines) and any("id:" in ln for ln in lines) and any("name:" in ln for ln in lines):
        current_id = None
        current_name = None
        for ln in lines:
            ln_low = ln.lower()
            if "id:" in ln_low:
                try:
                    current_id = int(ln.split(":")[1].strip().strip('"').strip("'"))
                except Exception:
                    current_id = None
            elif "name:" in ln_low:
                # Extract the name value after ':', strip quotes
                name_part = ln.split(":", 1)[1].strip()
                current_name = name_part.strip('"').strip("'")
            elif ln_low.startswith("item"):
                current_id = None
                current_name = None

            if current_id is not None and current_name is not None:
                labels[current_id] = current_name
                current_id = None
                current_name = None

        if labels:
            return labels

    # Fallback: simple formats like "0 person" or just "person" per line
    tmp = {}
    for idx, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                tmp[int(parts[0])] = " ".join(parts[1:])
            except Exception:
                tmp[idx] = ln
        else:
            tmp[idx] = ln
    return tmp


# =========================
# TFLite + EdgeTPU
# =========================
def make_interpreter(model_path):
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(EDGETPU_LIB)]
    )
    interpreter.allocate_tensors()
    return interpreter


def get_input_details(interpreter):
    details = interpreter.get_input_details()[0]
    ih, iw = details["shape"][1], details["shape"][2]
    dtype = details["dtype"]
    return iw, ih, dtype, details["index"]


def set_input(interpreter, input_index, tensor):
    interpreter.set_tensor(input_index, tensor)


# =========================
# Preprocessing
# =========================
def preprocess_frame(frame_bgr, input_w, input_h, input_dtype):
    # Convert BGR to RGB and resize to model input size
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h))

    if input_dtype == np.uint8:
        input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    else:
        # Normalize to [0,1] float32 if model expects float
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return input_tensor


# =========================
# Inference Output Parsing
# =========================
def _squeeze_safe(arr):
    try:
        return np.squeeze(arr)
    except Exception:
        return arr


def _safe_get(arr, idx, default_val):
    if arr is None:
        return default_val
    a = arr
    if np.ndim(a) == 0:
        try:
            return a.item()
        except Exception:
            return default_val
    length = a.shape[0]
    if length == 0:
        return default_val
    if idx < length:
        return a[idx]
    return a[-1]


def parse_detections(interpreter, threshold, img_w, img_h):
    # Robustly parse various TFLite detection head formats; avoid index errors
    output_details = interpreter.get_output_details()
    boxes = None
    scores = None
    classes = None
    num_dets = None

    # First pass: use names if present
    for od in output_details:
        name = od.get("name", "").lower()
        tensor = _squeeze_safe(interpreter.get_tensor(od["index"]))
        if "boxes" in name or "box" in name:
            boxes = tensor
        elif "scores" in name or "score" in name:
            scores = tensor
        elif "classes" in name or "class" in name:
            classes = tensor
        elif "num" in name and "detection" in name:
            # num_detections might be float; cast later
            num_dets = tensor

    # Second pass: heuristic if any are missing
    outs = [_squeeze_safe(interpreter.get_tensor(od["index"])) for od in output_details]
    if boxes is None:
        cand = [o for o in outs if (o is not None and o.ndim >= 1 and (o.shape[-1] == 4))]
        if cand:
            boxes = cand[0]
    if scores is None:
        # scores often 1D and length matches boxes count
        cand = [o for o in outs if (o is not None and o.ndim == 1 and o.dtype != np.object_)]
        if cand:
            # Prefer array that matches boxes length if possible
            if boxes is not None and boxes.ndim >= 2:
                blen = boxes.shape[0]
                best = None
                for c in cand:
                    if c.shape[0] == blen:
                        best = c
                        break
                scores = best if best is not None else cand[0]
            else:
                scores = cand[0]
    if classes is None:
        cand = [o for o in outs if (o is not None and o.ndim == 1 and (np.issubdtype(o.dtype, np.integer) or np.issubdtype(o.dtype, np.floating)))]
        if cand:
            # Try to match scores length
            if scores is not None and scores.ndim == 1:
                slen = scores.shape[0]
                best = None
                for c in cand:
                    if c.shape[0] == slen:
                        best = c
                        break
                classes = best if best is not None else cand[-1]
            else:
                classes = cand[-1]

    # Normalize shapes: boxes -> (N,4), scores/classes -> (N,)
    if boxes is not None:
        if boxes.ndim == 1 and boxes.size == 4:
            boxes = boxes.reshape((1, 4))
        elif boxes.ndim > 2:
            # If shape like (1, N, 4), squeeze first dim
            boxes = np.reshape(boxes, (-1, 4))
    if scores is not None and scores.ndim > 1:
        scores = scores.reshape(-1)
    if classes is not None and classes.ndim > 1:
        classes = classes.reshape(-1)

    # num_detections handling
    if num_dets is not None:
        try:
            num_val = int(np.round(float(_squeeze_safe(num_dets))))
        except Exception:
            num_val = None
    else:
        num_val = None

    # Decide number of detections robustly
    candidates = []
    if boxes is not None and boxes.ndim == 2:
        candidates.append(boxes.shape[0])
    if scores is not None and scores.ndim == 1:
        candidates.append(scores.shape[0])
    if classes is not None and classes.ndim == 1:
        candidates.append(classes.shape[0])
    if num_val is not None:
        candidates.append(num_val)
    n = min(candidates) if candidates else 0
    if n <= 0:
        return []

    detections = []
    # Determine if boxes are normalized (0..1) or absolute
    def to_pixel_box(b):
        y_min, x_min, y_max, x_max = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        if max(abs(y_min), abs(x_min), abs(y_max), abs(x_max)) <= 1.5:
            # normalized
            x1 = int(max(0, min(img_w - 1, x_min * img_w)))
            y1 = int(max(0, min(img_h - 1, y_min * img_h)))
            x2 = int(max(0, min(img_w - 1, x_max * img_w)))
            y2 = int(max(0, min(img_h - 1, y_max * img_h)))
        else:
            x1 = int(max(0, min(img_w - 1, x_min)))
            y1 = int(max(0, min(img_h - 1, y_min)))
            x2 = int(max(0, min(img_w - 1, x_max)))
            y2 = int(max(0, min(img_h - 1, y_max)))
        # Ensure proper ordering
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return x1, y1, x2, y2

    for i in range(n):
        score = float(_safe_get(scores, i, 0.0)) if scores is not None else 0.0
        if score < threshold:
            continue
        # Use class id if available; handle short arrays safely
        cls_val = _safe_get(classes, i, -1) if classes is not None else -1
        try:
            cls_id = int(round(float(cls_val)))
        except Exception:
            cls_id = -1

        if boxes is None:
            continue
        if i < boxes.shape[0]:
            box = boxes[i]
        else:
            box = boxes[-1]
        x1, y1, x2, y2 = to_pixel_box(box)
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": cls_id
        })

    return detections


# =========================
# Drawing and Metrics
# =========================
def draw_detections(frame_bgr, detections, labels, map_value=None):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cid = det["class_id"]
        label_text = labels.get(cid, f"id:{cid}") if labels else f"id:{cid}"
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label_text} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 8)
        cv2.rectangle(frame_bgr, (x1, ty - th - 4), (x1 + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if map_value is not None:
        map_text = f"mAP: {map_value:.3f}"
        cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 220), 2, cv2.LINE_AA)


def update_map_stats(stats, detections):
    # Stats: dict[class_id] -> list of detection confidences
    for det in detections:
        cid = det["class_id"]
        sc = float(det["score"])
        if cid not in stats:
            stats[cid] = []
        stats[cid].append(sc)


def compute_naive_map(stats):
    # NOTE: Without ground truth, true mAP cannot be computed.
    # Here we use a naive proxy: AP for a class = mean(confidences of its detections).
    # mAP = mean of per-class AP for classes that had any detections.
    if not stats:
        return 0.0
    aps = []
    for cid, scores in stats.items():
        if len(scores) == 0:
            continue
        aps.append(float(np.mean(scores)))
    if not aps:
        return 0.0
    return float(np.mean(aps))


# =========================
# Main Pipeline
# =========================
def main():
    # Prepare output directory
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Set up TFLite EdgeTPU interpreter
    interpreter = make_interpreter(MODEL_PATH)
    in_w, in_h, in_dtype, in_index = get_input_details(interpreter)

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open input video: {INPUT_PATH}")
        return

    # Prepare writer
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 0 or np.isnan(in_fps):
        in_fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        # Fallback if metadata not available
        ret, test_frame = cap.read()
        if not ret:
            print("Error: unable to read frames from input video.")
            cap.release()
            return
        frame_h, frame_w = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: cannot open video writer for: {OUTPUT_PATH}")
        cap.release()
        return

    # Metrics
    map_stats = {}
    last_log = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            input_tensor = preprocess_frame(frame, in_w, in_h, in_dtype)
            set_input(interpreter, in_index, input_tensor)

            # Inference
            interpreter.invoke()

            # Parse detections robustly (avoid index errors as seen previously)
            detections = parse_detections(interpreter, CONFIDENCE_THRESHOLD, frame_w, frame_h)

            # Update metrics (naive proxy since ground truth is unavailable)
            update_map_stats(map_stats, detections)
            curr_map = compute_naive_map(map_stats)

            # Draw results and mAP on frame
            draw_detections(frame, detections, labels, map_value=curr_map)

            # Write to output
            writer.write(frame)

            # Optional simple logging every few seconds
            if time.time() - last_log > 5.0:
                print(f"Processed {frame_count} frames; current naive mAP={curr_map:.3f}")
                last_log = time.time()
    finally:
        cap.release()
        writer.release()

    print(f"Done. Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()