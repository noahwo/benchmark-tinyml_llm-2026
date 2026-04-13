#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single input video using a TFLite SSD model,
draws bounding boxes and labels on detected objects, and saves an annotated output video.
It also calculates a proxy mAP (mean Average Precision) over the processed video by
treating Non-Maximum Suppressed (NMS) detections as pseudo ground truth and evaluating
raw model proposals against them.

Phases implemented following the provided Programming Guideline:
- Phase 1: Setup (imports, paths, load labels, load interpreter, get model details)
- Phase 2: Input Acquisition & Preprocessing Loop (read video frames, preprocess)
- Phase 3: Inference (set input tensors and invoke)
- Phase 4: Output Interpretation & Handling Loop
    4.1 Get output tensors
    4.2 Interpret results (map indices to labels, structure detections)
    4.3 Post-processing (thresholding, NMS, coordinate scaling/clipping)
    4.4 Handle output (draw and write annotated frames, compute and overlay mAP)
    4.5 Loop continuation
- Phase 5: Cleanup (release video resources)
"""

# =========================
# Phase 1: Setup
# =========================
# 1.1 Imports
import os
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESH = float('0.5')

# Utility: Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1.3 Load Labels
def load_labels(path):
    labels = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
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

# Extract input properties (assuming single input tensor)
in_shape = input_details[0]['shape']  # e.g., [1, H, W, 3]
in_height, in_width = int(in_shape[1]), int(in_shape[2])
in_dtype = input_details[0]['dtype']

# Helper: Determine which output tensor corresponds to boxes/scores/classes/num
def map_output_indices(interpreter, output_details):
    """
    After a first inference, map output tensor indices for:
    - boxes: shape [1, N, 4]
    - classes: shape [1, N]
    - scores: shape [1, N]
    - num_detections: scalar (1) or [1]
    """
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    boxes_idx = classes_idx = scores_idx = num_idx = None

    # Identify boxes by last dim == 4
    for i, od in enumerate(output_details):
        shp = od['shape']
        if len(shp) == 3 and shp[-1] == 4:
            boxes_idx = i
            break

    # Identify num_detections by being scalar or having single element
    for i, od in enumerate(output_details):
        shp = od['shape']
        total_elems = int(np.prod(shp))
        if total_elems == 1:
            num_idx = i
            break

    # Identify classes and scores among remaining [1, N]
    # Heuristic: classes have values generally > 1, scores within [0,1]
    cand = []
    for i, od in enumerate(output_details):
        if i in [boxes_idx, num_idx]:
            continue
        shp = od['shape']
        if len(shp) == 2 and shp[0] == 1:
            cand.append(i)

    # If both present, decide by value ranges after a dry inference
    if len(cand) == 2:
        a = outputs[cand[0]].flatten()
        b = outputs[cand[1]].flatten()
        # Decide which looks like scores (max <= 1.0) vs classes
        def is_scores(arr):
            if arr.size == 0:
                return True
            mx = np.nanmax(arr)
            mn = np.nanmin(arr)
            return (mx <= 1.0001 and mn >= 0.0)
        if is_scores(a) and not is_scores(b):
            scores_idx, classes_idx = cand[0], cand[1]
        elif is_scores(b) and not is_scores(a):
            scores_idx, classes_idx = cand[1], cand[0]
        else:
            # Fallback: assign arbitrarily, will still work for many models
            scores_idx, classes_idx = cand[0], cand[1]
    elif len(cand) == 1:
        # Some models may skip num_detections; assume the remaining is scores, and classes absent (unlikely)
        scores_idx = cand[0]
        classes_idx = None
    else:
        # Fallback, though standard SSD should have both
        scores_idx = classes_idx = None

    return boxes_idx, classes_idx, scores_idx, num_idx

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# Video capture setup
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

# Retrieve original video properties
orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 25.0  # default fallback if FPS not available

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_width, orig_height))
if not out_writer.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

# Helper functions for post-processing and metrics

def preprocess_frame(frame_bgr, target_w, target_h, dtype, floating_model):
    # Resize and convert BGR to RGB as typical for TFLite SSD models
    resized = cv2.resize(frame_bgr, (target_w, target_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = input_data.astype(dtype, copy=False)
    return input_data

def to_abs_box(box, img_w, img_h):
    # box: [ymin, xmin, ymax, xmax] normalized [0,1]
    ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    x1 = int(np.clip(xmin * img_w, 0, img_w - 1))
    y1 = int(np.clip(ymin * img_h, 0, img_h - 1))
    x2 = int(np.clip(xmax * img_w, 0, img_w - 1))
    y2 = int(np.clip(ymax * img_h, 0, img_h - 1))
    # Ensure proper ordering
    x1, x2 = (x1, x2) if x2 >= x1 else (x2, x1)
    y1, y2 = (y1, y2) if y2 >= y1 else (y2, y1)
    return [x1, y1, x2, y2]

def iou_xyxy(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    inter = inter_w * inter_h
    areaA = max(0, boxA[2] - boxA[0] + 1) * max(0, boxA[3] - boxA[1] + 1)
    areaB = max(0, boxB[2] - boxB[0] + 1) * max(0, boxB[3] - boxB[1] + 1)
    denom = float(areaA + areaB - inter + 1e-12)
    return inter / denom

def nms_per_class(boxes, scores, iou_thresh=0.5):
    # Greedy NMS for a single class
    if len(boxes) == 0:
        return []
    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_arr.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def class_color(cid):
    # Deterministic color per class id (BGR)
    return (int((cid * 37) % 256), int((cid * 17) % 256), int((cid * 29) % 256))

def get_label_name(cid):
    if labels and 0 <= cid < len(labels):
        return labels[cid]
    return f"class_{cid}"

def compute_map(preds_per_class, gts_per_class, iou_threshold=0.5):
    """
    Compute mAP over all classes present in gts_per_class using 11-point interpolation.
    preds_per_class: dict[cid] -> list of dicts { 'image_id': int, 'box': [x1,y1,x2,y2], 'score': float }
    gts_per_class:   dict[cid] -> list of dicts { 'image_id': int, 'box': [x1,y1,x2,y2] }
    """
    aps = []
    for cid in sorted(gts_per_class.keys()):
        gts = gts_per_class.get(cid, [])
        preds = preds_per_class.get(cid, [])

        if len(gts) == 0:
            continue

        # Organize GTs by image and track "matched" flags
        gts_by_img = {}
        for gt in gts:
            img_id = gt['image_id']
            gts_by_img.setdefault(img_id, [])
            gts_by_img[img_id].append({'box': gt['box'], 'matched': False})

        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda d: d['score'], reverse=True)

        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)
        total_gts = len(gts)

        for i, p in enumerate(preds_sorted):
            img_id = p['image_id']
            p_box = p['box']
            candidates = gts_by_img.get(img_id, [])
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(candidates):
                if gt['matched']:
                    continue
                current_iou = iou_xyxy(p_box, gt['box'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_j = j
            if best_iou >= iou_threshold and best_j >= 0:
                # True positive
                candidates[best_j]['matched'] = True
                tp[i] = 1.0
                fp[i] = 0.0
            else:
                # False positive
                tp[i] = 0.0
                fp[i] = 1.0

        if total_gts == 0:
            continue

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        recall = cum_tp / float(total_gts)

        # 11-point interpolated AP
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            precisions_at_r = precision[recall >= r]
            p_max = np.max(precisions_at_r) if precisions_at_r.size > 0 else 0.0
            ap += p_max
        ap /= 11.0
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# Prepare containers for mAP computation across the video
from collections import defaultdict
preds_per_class = defaultdict(list)  # cid -> list of {image_id, box, score}
gts_per_class = defaultdict(list)    # cid -> list of {image_id, box}

# Prepare floating model flag
floating_model = (in_dtype == np.float32)

# For mapping output indices, we need a one-time "dry run" or do it on first frame
mapped = False
boxes_idx = classes_idx = scores_idx = num_idx = None

# =========================
# Processing Loop
# =========================
frame_index = 0
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    # 2.2 Preprocess Data
    input_data = preprocess_frame(frame_bgr, in_width, in_height, in_dtype, floating_model)

    # 2.3 Quantization Handling already included in preprocess_frame per floating_model

    # =========================
    # Phase 3: Inference
    # =========================
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # =========================
    # Phase 4: Output Interpretation & Handling
    # =========================
    # 4.1 Get Output Tensors
    if not mapped:
        boxes_idx, classes_idx, scores_idx, num_idx = map_output_indices(interpreter, output_details)
        mapped = True

    # Retrieve raw outputs
    boxes_raw = interpreter.get_tensor(output_details[boxes_idx]['index'])  # [1, N, 4]
    boxes_raw = boxes_raw[0] if boxes_raw.ndim == 3 else boxes_raw

    # Some models include num_detections; otherwise derive from outputs length
    if num_idx is not None:
        num_dets_raw = interpreter.get_tensor(output_details[num_idx]['index'])
        num_dets = int(np.squeeze(num_dets_raw).astype(np.int32))
        if num_dets <= 0 or num_dets > boxes_raw.shape[0]:
            num_dets = boxes_raw.shape[0]
    else:
        num_dets = boxes_raw.shape[0]

    scores_raw = interpreter.get_tensor(output_details[scores_idx]['index']) if scores_idx is not None else None
    classes_raw = interpreter.get_tensor(output_details[classes_idx]['index']) if classes_idx is not None else None

    scores_raw = scores_raw[0] if scores_raw is not None and scores_raw.ndim == 2 else scores_raw
    classes_raw = classes_raw[0] if classes_raw is not None and classes_raw.ndim == 2 else classes_raw

    if scores_raw is None:
        # If scores not provided (unlikely for SSD), create dummy confidences
        scores_raw = np.ones((boxes_raw.shape[0],), dtype=np.float32)
    if classes_raw is None:
        # If classes not provided (unlikely), assign all to a default class 0
        classes_raw = np.zeros((boxes_raw.shape[0],), dtype=np.int32)

    # Clip to num detections
    boxes_raw = boxes_raw[:num_dets]
    scores_raw = scores_raw[:num_dets].astype(np.float32)
    classes_raw = classes_raw[:num_dets].astype(np.int32)

    # Convert boxes to absolute pixel coordinates and clip to frame size
    abs_boxes = [to_abs_box(b, orig_width, orig_height) for b in boxes_raw]
    abs_scores = scores_raw.tolist()
    abs_classes = classes_raw.tolist()

    # 4.2 Interpret Results
    # Structure detections per class
    detections_per_class = {}
    for i in range(len(abs_boxes)):
        cid = int(abs_classes[i])
        s = float(abs_scores[i])
        box = abs_boxes[i]
        detections_per_class.setdefault(cid, {'boxes': [], 'scores': []})
        detections_per_class[cid]['boxes'].append(box)
        detections_per_class[cid]['scores'].append(s)

    # 4.3 Post-processing: apply confidence thresholding and NMS, clip boxes already done
    final_detections = []  # list of dicts for drawing and GT for mAP proxy
    for cid, data in detections_per_class.items():
        boxes_c = data['boxes']
        scores_c = data['scores']
        # First filter by confidence threshold for display / GT
        idxs_conf = [i for i, sc in enumerate(scores_c) if sc >= CONF_THRESH]
        boxes_thr = [boxes_c[i] for i in idxs_conf]
        scores_thr = [scores_c[i] for i in idxs_conf]
        if len(boxes_thr) == 0:
            continue
        # Apply NMS per class
        keep_idx = nms_per_class(boxes_thr, scores_thr, iou_thresh=0.5)
        for k in keep_idx:
            final_detections.append({
                'class_id': cid,
                'score': float(scores_thr[k]),
                'box': boxes_thr[k]
            })

    # Prepare data for mAP computation (proxy: predictions vs NMS results as pseudo GT)
    # Add ground truths (NMS results) for this frame
    for det in final_detections:
        gts_per_class[det['class_id']].append({
            'image_id': frame_index,
            'box': det['box']
        })

    # Add predictions (raw model outputs) for this frame using all proposals (score > 0)
    for i in range(len(abs_boxes)):
        cid = int(abs_classes[i])
        preds_per_class[cid].append({
            'image_id': frame_index,
            'box': abs_boxes[i],
            'score': float(abs_scores[i])
        })

    # Compute running mAP over seen frames
    running_map = compute_map(preds_per_class, gts_per_class, iou_threshold=0.5)

    # 4.4 Handle Output: draw rectangles and labels, overlay mAP; then write to video
    canvas = frame_bgr.copy()
    for det in final_detections:
        cid = det['class_id']
        sc = det['score']
        x1, y1, x2, y2 = det['box']
        color = class_color(cid)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_text = f"{get_label_name(cid)}: {sc:.2f}"
        # Put label background for readability
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay running mAP
    map_text = f"mAP: {running_map:.3f}"
    cv2.putText(canvas, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)

    out_writer.write(canvas)

    frame_index += 1

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
out_writer.release()

print(f"Processing complete.")
print(f"Input video: {INPUT_PATH}")
print(f"Annotated output saved to: {OUTPUT_PATH}")