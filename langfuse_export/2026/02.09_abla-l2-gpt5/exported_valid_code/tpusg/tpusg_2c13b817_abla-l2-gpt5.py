import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(path):
    """
    Loads labels from a text file.
    Supports common formats:
      - "id label"
      - "id: label"
      - "label" (index inferred by row order)
    Returns a dict: {int_id: label_text}
    """
    labels = {}
    try:
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
    except Exception as e:
        print("Failed to read label file:", e)
        return labels

    auto_index = 0
    for line in lines:
        # Try formats with explicit index first
        if ":" in line:
            # e.g., "0: person"
            parts = line.split(":", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            if left.isdigit():
                labels[int(left)] = right
                continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            idx = int(parts[0])
            name = " ".join(parts[1:])
            labels[idx] = name
            continue
        # Fallback: use sequential indices
        labels[auto_index] = line
        auto_index += 1
    return labels


def make_interpreter(model_path, edgetpu_lib):
    """
    Creates TFLite interpreter with EdgeTPU delegate if available.
    """
    delegates = []
    if os.path.exists(edgetpu_lib):
        try:
            delegates = [load_delegate(edgetpu_lib)]
        except Exception as e:
            print("Warning: Failed to load EdgeTPU delegate:", e)
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame_bgr, input_details):
    """
    Preprocesses a single BGR frame to match the model input tensor.
    Converts BGR to RGB, resizes to model input size, handles dtype/scale.
    """
    ih, iw = input_details['shape'][1], input_details['shape'][2]
    dtype = input_details['dtype']

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)

    if dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    else:
        inp = resized.astype(dtype)

    inp = np.expand_dims(inp, axis=0)
    return inp


def map_outputs(interpreter):
    """
    Robustly maps the four detection outputs: boxes, classes, scores, num_detections.
    Some models may not provide explicit names or consistent ordering.
    This function infers the correct tensors by inspecting shapes, dtypes, and value ranges.

    Returns:
      boxes: np.ndarray with shape [N, 4] in normalized coordinates (ymin, xmin, ymax, xmax)
      classes: np.ndarray with shape [N] (int)
      scores: np.ndarray with shape [N] (float)
      num: int (number of valid detections)
    """
    details = interpreter.get_output_details()
    outputs = []
    for det in details:
        arr = interpreter.get_tensor(det['index'])
        name = det.get('name', '')
        outputs.append((name, arr))

    # Identify candidates
    boxes = None
    num = None
    flat_candidates = []  # arrays with shape [1, N]
    for name, arr in outputs:
        a = np.array(arr)
        if a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] == 4:
            boxes = a[0]
        elif a.size == 1:
            num = int(a.flatten()[0])
        elif a.ndim == 2 and a.shape[0] == 1 and a.shape[1] >= 1:
            flat_candidates.append((name, a[0]))

    # Heuristic to separate scores vs classes among flat candidates
    scores = None
    classes = None

    def frac_between_0_1(x):
        if x.size == 0:
            return 0.0
        return float(np.mean((x >= 0.0) & (x <= 1.0)))

    def frac_integer_like(x):
        if x.size == 0:
            return 0.0
        return float(np.mean(np.isclose(x, np.round(x))))

    # First try by name keywords if present
    for name, arr in flat_candidates:
        lname = name.lower()
        if "score" in lname:
            scores = arr.astype(np.float32)
        elif "class" in lname:
            classes = arr.astype(np.int32)

    # If still ambiguous, use value-pattern heuristics
    if scores is None or classes is None:
        # Sort by "score-likeness": values mostly in [0,1]
        if len(flat_candidates) >= 1:
            # Score-like candidate
            fl_sorted = sorted(flat_candidates, key=lambda t: frac_between_0_1(t[1]), reverse=True)
            # Class-like candidate: prefer integer-like
            cl_sorted = sorted(flat_candidates, key=lambda t: (frac_integer_like(t[1]), -frac_between_0_1(t[1])), reverse=True)

            if scores is None:
                scores = fl_sorted[0][1].astype(np.float32)
            if classes is None:
                # pick best integer-like that is not the same array as scores (by identity)
                for cand_name, cand_arr in cl_sorted:
                    if not np.may_share_memory(cand_arr, scores):
                        classes = cand_arr.astype(np.int32)
                        break
                if classes is None:
                    # fallback to second best by score-likeness
                    if len(fl_sorted) > 1:
                        classes = fl_sorted[1][1].astype(np.int32)
                    else:
                        classes = fl_sorted[0][1].astype(np.int32)

    # Final fallbacks
    if boxes is None:
        # Try to find any [N,4] or [1,N,4]
        for _, a in outputs:
            if a.ndim == 2 and a.shape[-1] == 4:
                boxes = a
                break
        if boxes is None:
            raise RuntimeError("Could not find detection boxes output tensor.")

    if num is None:
        # derive from length of scores or classes or boxes
        if scores is not None:
            num = int(scores.shape[0])
        elif classes is not None:
            num = int(classes.shape[0])
        else:
            num = int(boxes.shape[0])

    # Ensure shapes and dtypes
    boxes = np.array(boxes).reshape(-1, 4).astype(np.float32)
    classes = np.array(classes).reshape(-1).astype(np.int32) if classes is not None else np.zeros((boxes.shape[0],), dtype=np.int32)
    scores = np.array(scores).reshape(-1).astype(np.float32) if scores is not None else np.ones((boxes.shape[0],), dtype=np.float32)

    # Clip to provided num
    num = max(0, min(num, boxes.shape[0], classes.shape[0], scores.shape[0]))
    boxes = boxes[:num]
    classes = classes[:num]
    scores = scores[:num]

    return boxes, classes, scores, num


def draw_detections(frame_bgr, boxes, classes, scores, labels, threshold, running_map):
    """
    Draws bounding boxes and labels on the BGR frame for detections above threshold.
    Also overlays a running "mAP" score (proxy: mean confidence of kept detections so far).
    """
    h, w = frame_bgr.shape[:2]
    color = (0, 255, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        cls_id = int(classes[i])
        label = labels.get(cls_id, str(cls_id))

        # boxes are normalized ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = boxes[i]
        xmin = max(0, min(w - 1, int(xmin * w)))
        xmax = max(0, min(w - 1, int(xmax * w)))
        ymin = max(0, min(h - 1, int(ymin * h)))
        ymax = max(0, min(h - 1, int(ymax * h)))

        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, thickness)
        text = f"{label}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_bg_tl = (xmin, max(0, ymin - th - 6))
        text_bg_br = (xmin + tw + 6, max(0, ymin))
        cv2.rectangle(frame_bgr, text_bg_tl, text_bg_br, color, -1)
        cv2.putText(frame_bgr, text, (xmin + 3, ymin - 4), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Overlay running "mAP" (proxy: mean of all kept confidences so far)
    map_text = f"mAP: {running_map:.3f}"
    (mw, mh), _ = cv2.getTextSize(map_text, font, 0.6, 2)
    pad = 6
    cv2.rectangle(frame_bgr, (5, 5), (5 + mw + 2 * pad, 5 + mh + 2 * pad), (0, 0, 0), -1)
    cv2.putText(frame_bgr, map_text, (5 + pad, 5 + mh + pad), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    # Setup interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_LIB)
    input_details = interpreter.get_input_details()[0]

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        # Try to read one frame to infer size
        ret_probe, frame_probe = cap.read()
        if not ret_probe:
            cap.release()
            raise RuntimeError("Failed to read a frame to infer video size.")
        height, width = frame_probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    running_score_sum = 0.0
    running_score_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            inp = preprocess_frame(frame, input_details)

            # Inference
            interpreter.set_tensor(input_details['index'], inp)
            interpreter.invoke()

            # Postprocess/mapping
            boxes, classes, scores, num = map_outputs(interpreter)

            # Filter and accumulate scores for mAP proxy
            valid_mask = scores >= CONF_THRESHOLD
            kept_scores = scores[valid_mask]
            if kept_scores.size > 0:
                running_score_sum += float(np.sum(kept_scores))
                running_score_count += int(kept_scores.size)
            running_map = (running_score_sum / running_score_count) if running_score_count > 0 else 0.0

            # Draw
            draw_detections(frame, boxes, classes, scores, labels, CONF_THRESHOLD, running_map)

            # Write
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    # Optionally, print final mAP proxy
    final_map = (running_score_sum / running_score_count) if running_score_count > 0 else 0.0
    print(f"Processing completed. Saved to: {OUTPUT_PATH}")
    print(f"Final mAP (proxy based on mean confidence above threshold {CONF_THRESHOLD}): {final_map:.4f}")


if __name__ == "__main__":
    main()