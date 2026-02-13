#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single input video using a TFLite SSD model
and writes an annotated output video with bounding boxes and labels. It also computes
a running proxy mAP (mean Average Precision) using pseudo ground truth derived from
high-confidence detections (>= 0.8) and evaluates predictions at the configured
confidence threshold (>= 0.5) at IoU=0.5.

Notes:
- Interpreter is imported exactly as required: from ai_edge_litert.interpreter import Interpreter
- Only standard Python libraries and cv2/numpy are used.
- Phases 2, 4.2, and 4.3 are explicitly implemented as per the Programming Guidelines.

Proxy mAP explanation (due to lack of ground-truth annotations):
- Pseudo-GT is formed from detections with confidence >= 0.8, with per-class NMS to reduce duplicates.
- Predictions are detections with confidence >= 0.5 (as configured).
- For each class, predictions are matched to pseudo-GT by IoU >= 0.5 using a greedy strategy.
- AP is computed as area under precision-recall curve; mAP is the mean AP across classes that have at least 1 pseudo-GT instance.
"""

import os
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports: Interpreter must be imported literally as below
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (from configuration)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD  = 0.5
PSEUDO_GT_THRESHOLD = 0.8            # pseudo ground-truth threshold for mAP
IOU_THRESHOLD = 0.5                  # IoU threshold for evaluation and NMS

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def get_label_name(labels, class_id):
    """
    Attempts to map model class_id to a label name robustly.
    Many SSD MobileNet TFLite models use 0-based class indices; some use 1-based.
    """
    idx = int(class_id)
    # Prefer 0-based
    if 0 <= idx < len(labels):
        return labels[idx]
    # Try 1-based fallback
    idx1 = idx - 1
    if 0 <= idx1 < len(labels):
        return labels[idx1]
    return f'class_{int(class_id)}'

# 1.3 Load Labels
labels = load_labels(LABEL_PATH)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Retrieve input shape and dtype
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]
input_dtype = input_details[0]['dtype']
input_height = int(input_shape[1])
input_width = int(input_shape[2])
floating_model = (input_dtype == np.float32)

# =========================
# Utility Functions
# =========================

def preprocess_frame(frame_bgr, target_width, target_height, floating):
    """
    Resize and normalize the frame for model input.
    Converts BGR (OpenCV) to RGB as commonly expected by TFLite SSD models.
    Returns input_data shaped [1, H, W, 3] of dtype matching the model input.
    """
    img_resized = cv2.resize(frame_bgr, (target_width, target_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    if floating:
        input_data = img_rgb.astype(np.float32)
        input_data = (input_data - 127.5) / 127.5
    else:
        input_data = img_rgb.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def identify_outputs(interpreter, output_details):
    """
    Retrieve raw output tensors and reliably identify:
    - boxes: (1, N, 4)
    - classes: (1, N)
    - scores: (1, N)
    - num_detections: (1)
    Uses shape and value ranges to distinguish classes vs scores.
    """
    out_tensors = []
    for od in output_details:
        out_tensors.append(interpreter.get_tensor(od['index']))

    boxes = None
    classes = None
    scores = None
    num = None

    # First find boxes and num by shape
    for t in out_tensors:
        shp = t.shape
        if len(shp) == 3 and shp[-1] == 4:
            boxes = t
        elif len(shp) == 1 and shp[0] == 1:
            num = t

    # Remaining two are classes and scores
    cand_1d = [t for t in out_tensors if len(t.shape) == 2]
    if len(cand_1d) == 2:
        a, b = cand_1d[0], cand_1d[1]
        # Determine which is scores based on value ranges [0,1]
        a_vals = a.flatten()
        b_vals = b.flatten()
        a_is_scores = np.mean((a_vals >= 0.0) & (a_vals <= 1.0)) > 0.9
        b_is_scores = np.mean((b_vals >= 0.0) & (b_vals <= 1.0)) > 0.9
        # Assign based on heuristic
        if a_is_scores and not b_is_scores:
            scores, classes = a, b
        elif b_is_scores and not a_is_scores:
            scores, classes = b, a
        else:
            # If ambiguous, choose by max value: classes likely have max > 1
            if np.max(a_vals) > np.max(b_vals):
                classes, scores = a, b
            else:
                classes, scores = b, a

    return boxes, classes, scores, num

def clip_box(box, w, h):
    """
    Clip [xmin, ymin, xmax, ymax] to image bounds.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    # Ensure proper ordering after clipping
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def iou_xyxy(boxA, boxB):
    """
    Intersection-over-Union for boxes in [xmin, ymin, xmax, ymax] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    inter_area = inter_w * inter_h
    areaA = max(0, (boxA[2] - boxA[0] + 1)) * max(0, (boxA[3] - boxA[1] + 1))
    areaB = max(0, (boxB[2] - boxB[0] + 1)) * max(0, (boxB[3] - boxB[1] + 1))
    denom = float(areaA + areaB - inter_area) if (areaA + areaB - inter_area) > 0 else 1.0
    return inter_area / denom

def nms_per_class(boxes, scores, iou_thr=0.5):
    """
    Non-maximum suppression for a single class.
    boxes: list of [xmin, ymin, xmax, ymax]
    scores: list of float
    Returns indices kept.
    """
    if len(boxes) == 0:
        return []

    idxs = np.argsort(-np.array(scores))
    keep = []
    boxes_arr = np.array(boxes, dtype=np.float32)

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = []
        for j in idxs[1:]:
            ious.append(iou_xyxy(boxes_arr[i], boxes_arr[j]))
        ious = np.array(ious)
        remain = np.where(ious <= iou_thr)[0]
        idxs = idxs[remain + 1]
    return keep

def compute_map_pseudo_gt(gts_by_class, preds_by_class, iou_thr=0.5):
    """
    Compute mAP across classes using pseudo ground truth.
    gts_by_class: dict[class_id] -> dict[frame_idx] -> list of boxes
    preds_by_class: dict[class_id] -> list of tuples (frame_idx, score, box)
    Returns (mAP, AP_per_class_dict, num_classes_contributing)
    """
    ap_dict = {}
    contributing = 0
    for cls_id in sorted(set(list(gts_by_class.keys()) + list(preds_by_class.keys()))):
        gt_frames = gts_by_class.get(cls_id, {})
        preds = preds_by_class.get(cls_id, [])
        # Count GT instances
        npos = sum(len(boxes) for boxes in gt_frames.values())
        if npos == 0:
            # No GT for this class; skip in mean
            ap_dict[cls_id] = 0.0
            continue

        contributing += 1
        # Sort predictions by descending score
        preds_sorted = sorted(preds, key=lambda x: -x[1])
        # Track matches per frame
        gt_matched = {fidx: np.zeros(len(gt_frames[fidx]), dtype=bool) for fidx in gt_frames}

        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        for i, (fidx, score, pbox) in enumerate(preds_sorted):
            if fidx not in gt_frames or len(gt_frames[fidx]) == 0:
                fp[i] = 1.0
                continue
            gt_boxes = gt_frames[fidx]
            ious = np.array([iou_xyxy(pbox, g) for g in gt_boxes], dtype=np.float32)
            max_iou_idx = int(np.argmax(ious))
            max_iou = float(ious[max_iou_idx])
            if max_iou >= iou_thr and not gt_matched[fidx][max_iou_idx]:
                tp[i] = 1.0
                gt_matched[fidx][max_iou_idx] = True
            else:
                fp[i] = 1.0

        # Precision-Recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / float(npos)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

        # AP as area under precision envelope
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        # Sum over recall steps
        ap = 0.0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        ap_dict[cls_id] = ap

    if contributing == 0:
        mAP = 0.0
    else:
        mAP = float(np.mean([ap_dict[c] for c in ap_dict if c in gts_by_class and sum(len(v) for v in gts_by_class[c].values()) > 0]))
    return mAP, ap_dict, contributing

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire Input Data: Read a single video file from the given input_path
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

# Get video properties
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 25.0  # fallback

# Prepare VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_width, orig_height))
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

print("Starting inference on video...")
print(f"Model: {MODEL_PATH}")
print(f"Labels: {LABEL_PATH}")
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")
print(f"Confidence threshold (predictions): {CONFIDENCE_THRESHOLD}")
print(f"Pseudo-GT threshold: {PSEUDO_GT_THRESHOLD}, IoU threshold: {IOU_THRESHOLD}")

# Data structures for proxy mAP computation
# gts_by_class: dict[class_id] -> dict[frame_idx] -> list of boxes
gts_by_class = {}
# preds_by_class: dict[class_id] -> list of tuples (frame_idx, score, box)
preds_by_class = {}

frame_index = 0
start_time = time.time()

# 2.4 Loop through frames
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    # 2.2 Preprocess Data (resize/normalize based on input_details; store as input_data)
    input_data = preprocess_frame(frame_bgr, input_width, input_height, floating_model)

    # =========================
    # Phase 3: Inference
    # =========================
    # 3.1 Set Input Tensor
    interpreter.set_tensor(input_index, input_data)
    # 3.2 Run Inference
    interpreter.invoke()

    # =========================
    # Phase 4: Output Interpretation & Handling Loop
    # =========================
    # 4.1 Get Output Tensors
    boxes_raw, classes_raw, scores_raw, num_raw = identify_outputs(interpreter, output_details)

    # Extract and reshape outputs
    if boxes_raw is None or classes_raw is None or scores_raw is None:
        # Clean up resources before raising
        cap.release()
        writer.release()
        raise RuntimeError("Failed to identify output tensors (boxes/classes/scores).")

    boxes = np.squeeze(boxes_raw)       # shape (N, 4) normalized [ymin, xmin, ymax, xmax]
    classes = np.squeeze(classes_raw)   # shape (N,)
    scores = np.squeeze(scores_raw)     # shape (N,)
    if num_raw is not None:
        num_detections = int(np.squeeze(num_raw).astype(np.int32))
        boxes = boxes[:num_detections]
        classes = classes[:num_detections]
        scores = scores[:num_detections]

    # 4.2 Interpret Results: convert to absolute coords, map classes to labels
    detections = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score <= 0.0:
            continue
        # Decode normalized boxes to absolute pixel coords on original frame
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = int(xmin * orig_width)
        y1 = int(ymin * orig_height)
        x2 = int(xmax * orig_width)
        y2 = int(ymax * orig_height)
        # 4.3 Post-processing: clipping and thresholding later
        box_xyxy = clip_box([x1, y1, x2, y2], orig_width, orig_height)
        class_id = int(classes[i])
        label_name = get_label_name(labels, class_id)
        detections.append({
            'class_id': class_id,
            'label': label_name,
            'score': score,
            'box': box_xyxy
        })

    # 4.3 Post-processing: apply confidence thresholding for predictions; bounding box clipping already applied
    # Prepare predictions (>= CONFIDENCE_THRESHOLD) and pseudo-GT (>= PSEUDO_GT_THRESHOLD)
    # Pseudo-GT per class with NMS to reduce duplicates
    frame_pseudo_gt_by_class = {}
    for det in detections:
        cid = det['class_id']
        if det['score'] >= PSEUDO_GT_THRESHOLD:
            frame_pseudo_gt_by_class.setdefault(cid, {'boxes': [], 'scores': []})
            frame_pseudo_gt_by_class[cid]['boxes'].append(det['box'])
            frame_pseudo_gt_by_class[cid]['scores'].append(det['score'])

    # Apply per-class NMS for pseudo-GT
    for cid, val in frame_pseudo_gt_by_class.items():
        boxes_c = val['boxes']
        scores_c = val['scores']
        keep_idx = nms_per_class(boxes_c, scores_c, iou_thr=IOU_THRESHOLD)
        kept_boxes = [boxes_c[k] for k in keep_idx]
        # Update global GT dict
        gts_by_class.setdefault(cid, {})
        gts_by_class[cid][frame_index] = kept_boxes

    # Prepare predictions list per class
    for det in detections:
        if det['score'] >= CONFIDENCE_THRESHOLD:
            cid = det['class_id']
            preds_by_class.setdefault(cid, [])
            preds_by_class[cid].append((frame_index, det['score'], det['box']))

    # 4.2 Continue: draw detections that pass prediction threshold
    # Draw rectangle and label text on frame
    for det in detections:
        if det['score'] >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            score = det['score']
            # Draw bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put label and score
            text = f"{label}: {score:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame_bgr, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Compute running proxy mAP and overlay on the frame
    mAP_val, ap_dict, contributing_classes = compute_map_pseudo_gt(gts_by_class, preds_by_class, iou_thr=IOU_THRESHOLD)
    map_text = f"mAP (pseudo-GT@{PSEUDO_GT_THRESHOLD:.1f}, IoU@{IOU_THRESHOLD:.1f}): {mAP_val:.3f}"
    if contributing_classes == 0:
        map_text += " [insufficient pseudo-GT]"
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2, cv2.LINE_AA)

    # 4.4 Handle Output: write annotated frame to output video
    writer.write(frame_bgr)

    # 4.5 Loop Continuation
    frame_index += 1
    if frame_index % 30 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {frame_index} frames in {elapsed:.1f}s; Current proxy mAP: {mAP_val:.3f}")

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
writer.release()
total_time = time.time() - start_time

# Final mAP computation and reporting
final_mAP, final_ap_dict, contributing_classes = compute_map_pseudo_gt(gts_by_class, preds_by_class, iou_thr=IOU_THRESHOLD)
print("\nInference complete.")
print(f"Total frames processed: {frame_index}")
print(f"Total time: {total_time:.2f}s, FPS (processing): {frame_index / max(total_time, 1e-6):.2f}")
print(f"Final proxy mAP (pseudo-GT@{PSEUDO_GT_THRESHOLD:.1f}, IoU@{IOU_THRESHOLD:.1f}): {final_mAP:.4f}")
if contributing_classes == 0:
    print("Note: No pseudo ground truth instances were found; mAP is not meaningful.")
else:
    # Optionally, list AP for top few classes present
    present_classes = sorted([cid for cid in final_ap_dict if cid in gts_by_class and sum(len(v) for v in gts_by_class[cid].values()) > 0])
    print("Per-class AP (for classes with pseudo-GT):")
    for cid in present_classes[:10]:  # limit printout
        print(f"  {get_label_name(labels, cid)} (id {cid}): AP={final_ap_dict[cid]:.4f}")
print(f"Annotated video saved to: {OUTPUT_PATH}")