import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# ================================================================
# Application: Object Detection via a video file
# Target Device: Raspberry Pi 4B
# ================================================================

# =========================
# Phase 1: Setup
# =========================

# 1.2 Paths/Parameters
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5  # Confidence Threshold

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1.3 Load Labels (Conditional)
def load_labels(label_path):
    labels = []
    if os.path.isfile(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    return labels

labels = load_labels(LABEL_PATH)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Retrieve input tensor characteristics
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # [1, height, width, 3]
input_dtype = input_details[0]['dtype']
in_height, in_width = int(input_shape[1]), int(input_shape[2])

# Determine floating model
floating_model = (input_dtype == np.float32)

# =========================
# Utility Functions
# =========================

def preprocess_frame(frame_bgr, in_w, in_h, dtype, floating):
    # Resize and convert BGR -> RGB
    resized = cv2.resize(frame_bgr, (in_w, in_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if floating:
        # Normalize to [-1, 1]
        input_data = (np.float32(rgb) - 127.5) / 127.5
    else:
        input_data = np.asarray(rgb, dtype=dtype)
    return np.expand_dims(input_data, axis=0)

def clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2

def parse_detection_outputs(raw_outputs):
    """
    Identify indices of boxes, classes, scores, num_detections from raw output tensors.
    Returns dict with keys: 'boxes_idx', 'classes_idx', 'scores_idx', 'num_idx' (num_idx can be None).
    """
    boxes_idx = None
    classes_idx = None
    scores_idx = None
    num_idx = None

    # Identify indices by shape characteristics
    for i, arr in enumerate(raw_outputs):
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes_idx = i
        elif arr.size == 1:
            num_idx = i

    # Among remaining, distinguish classes vs scores
    # Typical: both have shape (1, N); classes are float of integers > 1, scores in [0,1]
    candidate_idxs = [i for i in range(len(raw_outputs)) if i not in [boxes_idx] + ([num_idx] if num_idx is not None else [])]
    for i in candidate_idxs:
        arr = raw_outputs[i]
        if arr.ndim == 2:
            flat = arr.reshape(-1)
            # Use value range heuristic
            if np.all(flat >= 0.0) and np.all(flat <= 1.0):
                scores_idx = i
            else:
                classes_idx = i

    # Fallback: if ambiguous, assign deterministically
    if scores_idx is None or classes_idx is None:
        for i in candidate_idxs:
            arr = raw_outputs[i]
            if arr.ndim == 2:
                flat = arr.reshape(-1)
                if scores_idx is None:
                    scores_idx = i
                elif classes_idx is None:
                    classes_idx = i

    return {
        'boxes_idx': boxes_idx,
        'classes_idx': classes_idx,
        'scores_idx': scores_idx,
        'num_idx': num_idx
    }

def scale_boxes_to_frame(norm_box, frame_w, frame_h):
    # TFLite SSD: boxes in [ymin, xmin, ymax, xmax], normalized [0,1]
    y_min, x_min, y_max, x_max = norm_box
    x1 = int(x_min * frame_w)
    y1 = int(y_min * frame_h)
    x2 = int(x_max * frame_w)
    y2 = int(y_max * frame_h)
    # clip to image bounds
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, frame_w, frame_h)
    return x1, y1, x2, y2

def iou_xyxy(a, b):
    # boxes: [x1, y1, x2, y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def evaluate_frame_predictions(prev_gt_by_class, detections_by_class, pred_stats, gt_counts, iou_thresh=0.5):
    """
    Update running prediction stats for mAP calculation using previous frame's detections as pseudo-GT.
    - prev_gt_by_class: dict[int, list of boxes]
    - detections_by_class: dict[int, list of (score, box)]
    - pred_stats: dict[int, list of (score, is_tp)]
    - gt_counts: dict[int, int]
    """
    for cls_id, gt_list in prev_gt_by_class.items():
        gt_counts[cls_id] = gt_counts.get(cls_id, 0) + len(gt_list)

    for cls_id, preds in detections_by_class.items():
        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        gt_list = prev_gt_by_class.get(cls_id, [])
        gt_matched = [False] * len(gt_list)

        for score, p_box in preds_sorted:
            best_iou = 0.0
            best_idx = -1
            for gi, g_box in enumerate(gt_list):
                if not gt_matched[gi]:
                    iou = iou_xyxy(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi
            is_tp = 0
            if best_iou >= iou_thresh and best_idx >= 0:
                is_tp = 1
                gt_matched[best_idx] = True
            pred_stats.setdefault(cls_id, []).append((score, is_tp))

def compute_map(pred_stats, gt_counts):
    """
    Compute mAP using VOC 2007 11-point interpolation across classes present in gt_counts > 0.
    pred_stats: dict[int, list of (score, is_tp)]
    gt_counts: dict[int, int]
    Returns mAP, per_class_AP dict
    """
    ap_per_class = {}
    for cls_id, gt_total in gt_counts.items():
        if gt_total <= 0:
            continue
        preds = pred_stats.get(cls_id, [])
        if not preds:
            ap_per_class[cls_id] = 0.0
            continue
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        tps = np.array([p[1] for p in preds_sorted], dtype=np.float32)
        fps = 1.0 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
        recall = tp_cum / float(gt_total)

        # 11-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            mask = recall >= r
            p = np.max(precision[mask]) if np.any(mask) else 0.0
            ap += p / 11.0
        ap_per_class[cls_id] = float(ap)

    valid_aps = list(ap_per_class.values())
    mAP = float(np.mean(valid_aps)) if len(valid_aps) > 0 else 0.0
    return mAP, ap_per_class

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire Input Data
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

# Retrieve properties for output writer
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
if not out_writer.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open output video writer: {OUTPUT_PATH}")

# Variables for mAP computation across frames (using previous frame as pseudo-GT)
prev_gt_by_class = {}  # dict: class_id -> list of boxes [[x1,y1,x2,y2], ...]
pred_stats = {}        # dict: class_id -> list of (score, is_tp)
gt_counts = {}         # dict: class_id -> int

# Helper for output mapping (initialized after first inference)
output_map = None

frame_index = 0
start_time = time.time()

# =========================
# Processing Loop
# =========================
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    # 2.2 Preprocess Data
    input_data = preprocess_frame(frame_bgr, in_width, in_height, input_dtype, floating_model)

    # 2.3 Quantization Handling already done in preprocess_frame based on floating_model

    # =========================
    # Phase 3: Inference
    # =========================
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # =========================
    # Phase 4: Output Interpretation & Handling
    # =========================

    # 4.1 Get Output Tensors (raw)
    raw_outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    # Initialize output mapping on first frame
    if output_map is None:
        output_map = parse_detection_outputs(raw_outputs)

    boxes_arr = raw_outputs[output_map['boxes_idx']]
    classes_arr = raw_outputs[output_map['classes_idx']]
    scores_arr = raw_outputs[output_map['scores_idx']]
    if output_map['num_idx'] is not None:
        num_det = int(np.squeeze(raw_outputs[output_map['num_idx']]).tolist())
    else:
        num_det = boxes_arr.shape[1] if boxes_arr.ndim >= 2 else boxes_arr.shape[0]

    # Squeeze to remove batch dimension
    boxes = np.squeeze(boxes_arr, axis=0)
    classes = np.squeeze(classes_arr, axis=0)
    scores = np.squeeze(scores_arr, axis=0)

    # 4.2 Interpret Results
    # Prepare detections filtered by confidence and scaled to frame size
    dets_for_drawing = []  # list of (class_id, score, (x1,y1,x2,y2))
    detections_by_class = {}  # class_id -> list of (score, [x1,y1,x2,y2])

    take_n = min(num_det, boxes.shape[0])
    for i in range(take_n):
        score = float(scores[i])
        if score < CONF_THRESHOLD:
            continue
        cls_id = int(classes[i])
        y_min, x_min, y_max, x_max = boxes[i]
        x1, y1, x2, y2 = scale_boxes_to_frame((y_min, x_min, y_max, x_max), frame_w, frame_h)
        # 4.3 Post-processing: bounding box clipping is already ensured in scale function
        dets_for_drawing.append((cls_id, score, (x1, y1, x2, y2)))
        detections_by_class.setdefault(cls_id, []).append((score, (x1, y1, x2, y2)))

    # Update mAP statistics using previous frame detections as pseudo-ground truth
    if frame_index > 0:
        evaluate_frame_predictions(prev_gt_by_class, detections_by_class, pred_stats, gt_counts, iou_thresh=0.5)

    # Prepare GT for next frame (current detections act as GT for next iteration)
    prev_gt_by_class = {cid: [box for (_, box) in items] for cid, items in detections_by_class.items()}

    # Compute running mAP
    mAP_value, _ = compute_map(pred_stats, gt_counts)

    # 4.4 Handle Output: Draw detections and write to output video
    for cls_id, score, (x1, y1, x2, y2) in dets_for_drawing:
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        # Resolve label
        if 0 <= cls_id < len(labels):
            cls_name = labels[cls_id]
        else:
            cls_name = str(cls_id)
        label_text = f"{cls_name}: {score:.2f}"
        cv2.putText(frame_bgr, label_text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Overlay running mAP on the frame
    cv2.putText(frame_bgr, f"mAP (proxy): {mAP_value:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Frame: {frame_index}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    out_writer.write(frame_bgr)
    frame_index += 1

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
out_writer.release()

elapsed = time.time() - start_time
final_mAP, per_class_ap = compute_map(pred_stats, gt_counts)

print("Processing complete.")
print(f"Frames processed: {frame_index}")
print(f"Elapsed time: {elapsed:.2f} s, FPS (avg): {frame_index / elapsed if elapsed > 0 else 0:.2f}")
print(f"Proxy mAP over video (using previous frame as pseudo-GT): {final_mAP:.4f}")

# Optionally, print AP per class for classes present in GT
if per_class_ap:
    print("Per-class AP (classes with pseudo-GT present):")
    for cls_id in sorted(per_class_ap.keys()):
        name = labels[cls_id] if 0 <= cls_id < len(labels) else str(cls_id)
        print(f" - {name} (id {cls_id}): {per_class_ap[cls_id]:.4f}")