import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"  # fixed closing quote
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# =========================
# Utility: Label loading
# =========================
def load_labels(label_path):
    """
    Loads labels from a file. Supports lines of the form:
    - "id label"
    - "id: label"
    - "label" (implicit incremental id starting from 0)
    Returns dict: {int_id: str_label}
    """
    labels = {}
    next_id = 0
    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    left, right = line.split(":", 1)
                    left = left.strip()
                    right = right.strip()
                    if left.isdigit():
                        labels[int(left)] = right
                    else:
                        labels[next_id] = line
                        next_id += 1
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        labels[int(parts[0])] = parts[1].strip()
                    else:
                        labels[next_id] = line
                        next_id += 1
    except Exception:
        # Fallback if label file cannot be read; use empty dict
        labels = {}
    return labels

# =========================
# Utility: IoU computation
# =========================
def compute_iou(box_a, box_b):
    """
    box format: [xmin, ymin, xmax, ymax]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union

# =========================
# Utility: AP per class (temporal proxy)
# =========================
def average_precision_for_class(gt_boxes, pred_boxes, pred_scores, iou_thresh=0.5):
    """
    Compute Average Precision (11-point interpolated) for a single class,
    using previous frame detections as pseudo-ground-truth (temporal proxy).
    - gt_boxes: list of [xmin, ymin, xmax, ymax]
    - pred_boxes: list of [xmin, ymin, xmax, ymax]
    - pred_scores: list of floats
    Returns AP (float) or None if no gt is available.
    """
    num_gt = len(gt_boxes)
    if num_gt == 0:
        return None

    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by descending score
    order = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in order]
    pred_scores = [pred_scores[i] for i in order]

    matched_gt = set()
    tp = np.zeros(len(pred_boxes), dtype=np.float32)
    fp = np.zeros(len(pred_boxes), dtype=np.float32)

    # Greedy matching
    for i, pbox in enumerate(pred_boxes):
        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pbox, gbox)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp[i] = 1.0
            matched_gt.add(best_j)
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / float(num_gt)

    # 11-point interpolated AP
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        # max precision where recall >= r
        mask = recall >= r
        p = np.max(precision[mask]) if np.any(mask) else 0.0
        ap += p
    ap /= 11.0
    return float(ap)

def temporal_proxy_map(prev_dets, curr_dets, iou_thresh=0.5):
    """
    Compute temporal proxy mAP between consecutive frames:
    Uses detections of previous frame as pseudo ground truth for the current frame.
    - prev_dets, curr_dets: list of dicts with keys: 'bbox' [xmin,ymin,xmax,ymax], 'class_id', 'score'
    Returns mean AP across classes present in previous frame, or None if undefined.
    """
    if prev_dets is None or len(prev_dets) == 0:
        return None

    # Group detections by class
    classes = set([d['class_id'] for d in prev_dets] + [d['class_id'] for d in curr_dets])
    ap_values = []

    for c in classes:
        gt_boxes = [d['bbox'] for d in prev_dets if d['class_id'] == c]
        pred_boxes = [d['bbox'] for d in curr_dets if d['class_id'] == c]
        pred_scores = [float(d['score']) for d in curr_dets if d['class_id'] == c]
        ap_c = average_precision_for_class(gt_boxes, pred_boxes, pred_scores, iou_thresh=iou_thresh)
        if ap_c is not None:
            ap_values.append(ap_c)

    if len(ap_values) == 0:
        return None
    return float(np.mean(ap_values))

# =========================
# Interpreter helpers
# =========================
def make_interpreter(model_path, edgetpu_lib):
    delegates = [load_delegate(edgetpu_lib)]
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    return interpreter

def get_input_size_dtype(interpreter):
    input_details = interpreter.get_input_details()[0]
    _, h, w, _ = input_details['shape']
    dtype = input_details['dtype']
    return (w, h), dtype

def set_input(interpreter, frame_bgr, input_size, input_dtype):
    """
    Preprocess: resize, convert BGR->RGB, set dtype, set input tensor.
    """
    iw, ih = input_size
    resized = cv2.resize(frame_bgr, (iw, ih))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        # Normalize to [0,1]
        tensor = (rgb.astype(np.float32) / 255.0).astype(np.float32)
    else:
        tensor = rgb.astype(np.uint8)
    tensor = np.expand_dims(tensor, axis=0)
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details['index'], tensor)

def run_inference(interpreter):
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    # Many EdgeTPU object detection models (SSD) have 4 outputs:
    # boxes, classes, scores, count
    # We assume the conventional ordering used by TFLite detection postprocess.
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    count = interpreter.get_tensor(output_details[3]['index'])
    if np.isscalar(count):
        num = int(count)
    else:
        num = int(count[0]) if count.size > 0 else len(scores)

    # Ensure proper dtypes
    classes = classes.astype(np.int32)
    scores = scores.astype(np.float32)
    boxes = boxes.astype(np.float32)

    # Clip length to 'num' if necessary
    boxes = boxes[:num]
    classes = classes[:num]
    scores = scores[:num]

    return boxes, classes, scores, num

def scale_and_clip_boxes(box, frame_w, frame_h):
    """
    Convert normalized [ymin, xmin, ymax, xmax] to pixel [xmin, ymin, xmax, ymax],
    and clip to frame bounds.
    """
    ymin, xmin, ymax, xmax = box
    xmin_px = int(max(0, min(frame_w - 1, xmin * frame_w)))
    xmax_px = int(max(0, min(frame_w - 1, xmax * frame_w)))
    ymin_px = int(max(0, min(frame_h - 1, ymin * frame_h)))
    ymax_px = int(max(0, min(frame_h - 1, ymax * frame_h)))
    # Ensure proper ordering
    xmin_px, xmax_px = min(xmin_px, xmax_px), max(xmin_px, xmax_px)
    ymin_px, ymax_px = min(ymin_px, ymax_px), max(ymin_px, ymax_px)
    return [xmin_px, ymin_px, xmax_px, ymax_px]

def color_for_class(class_id):
    # Deterministic color from class id
    np.random.seed(class_id + 42)
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return color

# =========================
# Main pipeline
# =========================
def main():
    # 1. Setup
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_SHARED_LIB)
    labels = load_labels(LABEL_PATH)

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Failed to open input video:", INPUT_PATH)
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 30.0  # fallback

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print("Failed to open output video writer:", OUTPUT_PATH)
        cap.release()
        return

    input_size, input_dtype = get_input_size_dtype(interpreter)

    # For mAP (temporal proxy) accumulation
    prev_dets = None
    map_sum = 0.0
    map_count = 0

    frame_index = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Preprocessing
        set_input(interpreter, frame, input_size, input_dtype)

        # 3. Inference
        boxes, classes, scores, num = run_inference(interpreter)

        # 4. Output handling: build detections list, draw, compute temporal proxy mAP, save
        detections = []
        # Build list of detections in original frame coordinates
        for i in range(num):
            score = float(scores[i])
            cls_id = int(classes[i])
            box_norm = boxes[i]  # [ymin, xmin, ymax, xmax] normalized
            bbox = scale_and_clip_boxes(box_norm, frame_w, frame_h)

            det = {
                "bbox": bbox,
                "class_id": cls_id,
                "score": score
            }
            detections.append(det)

        # Compute temporal proxy mAP vs previous frame detections
        curr_map = temporal_proxy_map(prev_dets, detections, iou_thresh=0.5)
        if curr_map is not None:
            map_sum += curr_map
            map_count += 1

        running_map_text = "mAP (temporal proxy): N/A" if map_count == 0 else f"mAP (temporal proxy): {map_sum / map_count:.3f}"

        # Draw detections above confidence threshold
        for det in detections:
            if det["score"] < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            score = det["score"]
            color = color_for_class(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = labels.get(cls_id, f"id:{cls_id}")
            caption = f"{label} {score:.2f}"
            # Put background for text for better visibility
            (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Put mAP text on the frame
        cv2.putText(frame, running_map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 200, 25), 2, cv2.LINE_AA)

        writer.write(frame)

        # Prepare for next iteration
        prev_dets = detections
        frame_index += 1

    elapsed = time.time() - t0
    final_map = (map_sum / map_count) if map_count > 0 else float('nan')

    # Cleanup
    cap.release()
    writer.release()

    # Console summary
    print("Processing completed.")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_index}")
    print(f"Elapsed time: {elapsed:.2f} s, FPS: {frame_index / max(elapsed, 1e-6):.2f}")
    if map_count > 0:
        print(f"Temporal proxy mAP over video: {final_map:.4f}")
    else:
        print("Temporal proxy mAP over video: N/A")

if __name__ == "__main__":
    main()