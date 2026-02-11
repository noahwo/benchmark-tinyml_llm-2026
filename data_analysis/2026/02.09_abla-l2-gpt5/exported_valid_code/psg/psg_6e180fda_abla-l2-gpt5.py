import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# =======================
# Configuration parameters
# =======================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5  # Only keep detections above this
# =======================


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_labels(label_path):
    """
    Load labels from a label map text file.
    Each non-empty, non-comment line is treated as a label.
    Index in the list corresponds to the class id.
    """
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Typical TF Lite label files are plain names per line
            labels.append(line)
    return labels


def build_interpreter(model_path):
    """
    Build and allocate a TFLite interpreter using ai_edge_litert.
    """
    interpreter = Interpreter(model_path=model_path, num_threads=max(1, (os.cpu_count() or 4)))
    interpreter.allocate_tensors()
    return interpreter


def get_input_size_and_type(interpreter):
    input_details = interpreter.get_input_details()[0]
    ih, iw = input_details["shape"][1], input_details["shape"][2]
    idtype = input_details["dtype"]
    return iw, ih, idtype, input_details["index"]


def preprocess_frame(frame_bgr, in_w, in_h, in_dtype):
    """
    Resize and format input frame for the model.
    SSD MobileNet expects RGB input of size (in_h, in_w).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if in_dtype == np.float32:
        tensor = resized.astype(np.float32) / 255.0
    else:
        tensor = resized.astype(np.uint8)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def _get_output_array_by_name_or_fallback(interpreter, keyword):
    """
    Try to get an output tensor by matching a keyword in its name.
    If not found, return None.
    """
    outputs = []
    output_details = interpreter.get_output_details()
    for od in output_details:
        name = (od.get("name") or "").lower()
        if keyword in name:
            arr = interpreter.get_tensor(od["index"])
            outputs.append(arr)
    if outputs:
        return outputs[0]
    return None


def extract_detections(interpreter):
    """
    Extracts boxes, classes, scores, and num_detections from interpreter outputs.
    Returns:
      boxes: (N, 4) array in normalized [ymin, xmin, ymax, xmax]
      classes: (N,) array of class indices (float or int)
      scores: (N,) array of confidence scores
    """
    # First, try by canonical names
    boxes = _get_output_array_by_name_or_fallback(interpreter, "boxes")
    classes = _get_output_array_by_name_or_fallback(interpreter, "classes")
    scores = _get_output_array_by_name_or_fallback(interpreter, "scores")
    num = _get_output_array_by_name_or_fallback(interpreter, "num")

    # If any are None, fallback to heuristic using shapes
    if boxes is None or classes is None or scores is None:
        output_details = interpreter.get_output_details()
        outputs = [interpreter.get_tensor(od["index"]) for od in output_details]

        # Identify boxes: last dim == 4 after squeeze
        cand_boxes = []
        cand_others = []
        for arr in outputs:
            sq = np.squeeze(arr)
            if sq.ndim == 2 and sq.shape[1] == 4:
                cand_boxes.append(sq)
            else:
                cand_others.append(sq)

        if boxes is None and cand_boxes:
            boxes = cand_boxes[0]

        # For the remaining, pick two 1D arrays of the same length as boxes for classes and scores
        if boxes is not None and (classes is None or scores is None):
            N = boxes.shape[0]
            one_d = [a for a in cand_others if a.ndim == 1 and a.shape[0] == N]
            # Heuristic: the array with more integer-like values is classes; the other is scores
            if len(one_d) >= 2:
                # Pick two arrays
                a, b = one_d[0], one_d[1]
                # Determine which seems more "score-like" by mean value range [0,1]
                def score_like(x):
                    return float(np.mean((x >= 0.0) & (x <= 1.0)))
                if score_like(a) >= score_like(b):
                    scores = a
                    classes = b
                else:
                    scores = b
                    classes = a

        # Try to get num detections if still missing
        if num is None:
            for arr in outputs:
                sq = np.squeeze(arr)
                if sq.ndim == 0:
                    num = sq
                    break

    # Final safety conversion and squeezing
    if boxes is not None:
        boxes = np.squeeze(boxes)
        if boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes[0]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)

    if classes is not None:
        classes = np.squeeze(classes)
    else:
        classes = np.zeros((0,), dtype=np.float32)

    if scores is not None:
        scores = np.squeeze(scores)
    else:
        scores = np.zeros((0,), dtype=np.float32)

    # Handle num_detections if provided
    if num is not None:
        try:
            n = int(np.squeeze(num).item())
        except Exception:
            n = None
    else:
        n = None

    # Normalize shapes: ensure 2D boxes, 1D classes/scores
    if boxes.ndim == 1 and boxes.size == 4:
        boxes = boxes.reshape(1, 4)
    if classes.ndim == 0:
        classes = classes.reshape(1)
    if scores.ndim == 0:
        scores = scores.reshape(1)

    # Truncate to the common minimum length to avoid indexing errors
    lengths = [boxes.shape[0], classes.shape[0], scores.shape[0]]
    if n is not None:
        lengths.append(n)
    N = int(min(lengths)) if lengths else 0

    boxes = boxes[:N] if boxes.shape[0] >= N else boxes
    classes = classes[:N] if classes.shape[0] >= N else classes
    scores = scores[:N] if scores.shape[0] >= N else scores

    return boxes, classes, scores


def filter_and_scale_detections(boxes, classes, scores, conf_thres, frame_w, frame_h):
    """
    Filter detections by confidence and convert normalized boxes to pixel coordinates.
    Returns lists: px_boxes [(x1,y1,x2,y2)], classes, scores
    """
    if boxes.size == 0 or scores.size == 0:
        return [], [], []

    keep = scores >= float(conf_thres)
    boxes = boxes[keep]
    classes = classes[keep] if len(classes) == len(keep) else classes[:boxes.shape[0]]
    scores = scores[keep]

    px_boxes = []
    out_classes = []
    out_scores = []

    for i in range(boxes.shape[0]):
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = int(max(0, min(frame_w - 1, round(xmin * frame_w))))
        y1 = int(max(0, min(frame_h - 1, round(ymin * frame_h))))
        x2 = int(max(0, min(frame_w - 1, round(xmax * frame_w))))
        y2 = int(max(0, min(frame_h - 1, round(ymax * frame_h))))
        # Ensure proper ordering
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        # Discard degenerate boxes
        if (x2 - x1) >= 1 and (y2 - y1) >= 1:
            px_boxes.append((x1, y1, x2, y2))
            out_classes.append(classes[i] if i < len(classes) else -1)
            out_scores.append(scores[i])

    return px_boxes, out_classes, out_scores


def class_color(cls_id):
    """
    Deterministic BGR color for a class id.
    """
    base = int(cls_id) if isinstance(cls_id, (int, np.integer)) else int(float(cls_id))
    b = (37 * base) % 255
    g = (17 * base) % 255
    r = (73 * base) % 255
    return (b, g, r)


def draw_detections_on_frame(frame, px_boxes, classes, scores, labels, conf_thres, map_estimate=None, fps=None):
    """
    Draw bounding boxes and labels on the frame.
    Also overlay an estimated mAP and FPS if provided.
    """
    h, w = frame.shape[:2]
    thickness = max(1, int(round(min(h, w) / 300)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(h, w) / 1000.0)
    text_thickness = max(1, int(round(thickness)))

    n = min(len(px_boxes), len(classes), len(scores))
    for i in range(n):
        x1, y1, x2, y2 = px_boxes[i]
        cls_id = int(classes[i]) if len(classes) > i else -1
        score = float(scores[i]) if len(scores) > i else 0.0

        color = class_color(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build label string
        if 0 <= cls_id < len(labels):
            label_text = labels[cls_id]
        else:
            label_text = f"id:{cls_id}"
        label = f"{label_text} {score:.2f}"

        # Text background
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, text_thickness)
        y_text = max(th + 4, y1 - 4)
        x_text = x1
        cv2.rectangle(frame, (x_text, y_text - th - 4), (x_text + tw + 4, y_text + 2), color, -1)
        cv2.putText(frame, label, (x_text + 2, y_text - 2), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    # Overlay mAP (estimated/proxy)
    overlay_y = 24
    if map_estimate is not None:
        map_text = f"mAP (proxy): {map_estimate:.3f}"
        cv2.putText(frame, map_text, (8, overlay_y), font, font_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)
        overlay_y += int(20 * font_scale + 8)

    # Overlay FPS
    if fps is not None and fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (8, overlay_y), font, font_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)

    return frame


def main():
    # Prepare output directory
    ensure_parent_dir(OUTPUT_PATH)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Build interpreter
    interpreter = build_interpreter(MODEL_PATH)
    in_w, in_h, in_dtype, in_index = get_input_size_and_type(interpreter)

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3 or np.isnan(fps):
        fps = 30.0  # Fallback if FPS is unavailable
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        # Read one frame to infer size
        ok, test_frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Unable to read a frame from the input video to determine size.")
        height, width = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # For proxy mAP calculation: we'll use mean of detection scores above threshold across all frames
    all_kept_scores = []

    frame_count = 0
    t0 = time.time()
    last_time = t0
    fps_smooth = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            input_tensor = preprocess_frame(frame, in_w, in_h, in_dtype)

            # Inference
            interpreter.set_tensor(in_index, input_tensor)
            interpreter.invoke()

            # Extract detections
            boxes, classes, scores = extract_detections(interpreter)

            # Filter and scale detections
            px_boxes, cls_list, scr_list = filter_and_scale_detections(
                boxes, classes, scores, CONFIDENCE_THRESHOLD, frame_w=width, frame_h=height
            )

            # Update proxy mAP statistics
            if len(scr_list) > 0:
                all_kept_scores.extend([float(s) for s in scr_list])

            map_proxy = float(np.mean(all_kept_scores)) if all_kept_scores else 0.0

            # FPS calculation (smoothed)
            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            if fps_smooth is None:
                fps_smooth = inst_fps
            else:
                fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

            # Draw and write
            drawn = draw_detections_on_frame(
                frame, px_boxes, cls_list, scr_list, labels, CONFIDENCE_THRESHOLD, map_estimate=map_proxy, fps=fps_smooth
            )
            writer.write(drawn)

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    final_fps = frame_count / total_time if total_time > 0 else 0.0
    final_map_proxy = float(np.mean(all_kept_scores)) if all_kept_scores else 0.0

    print("Processing complete.")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {final_fps:.2f}")
    print(f"Estimated mAP (proxy using mean confidence over threshold {CONFIDENCE_THRESHOLD}): {final_map_proxy:.4f}")


if __name__ == "__main__":
    main()