#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection using a TFLite SSD model on a single input video file,
draws bounding boxes and labels on detected objects, writes the annotated video to an output path,
and computes a proxy mAP (mean Average Precision) based on temporal consistency between consecutive frames.

Programming Guideline Phases are explicitly followed and annotated in the code.
"""

# =========================
# Phase 1: Setup
# =========================

import os
import time
import numpy as np
import cv2

# 1.1 Imports: Import interpreter literally by the specified path
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5  # Confidence Threshold

# Utility: Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1.3 Load Labels (Conditional)
def load_labels(label_path):
    labels = []
    if os.path.isfile(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if len(name) > 0:
                    labels.append(name)
    return labels

labels = load_labels(LABEL_PATH)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assuming single input tensor for SSD models
input_shape = input_details[0]['shape']
input_height = int(input_shape[1])
input_width = int(input_shape[2])
input_dtype = input_details[0]['dtype']

# Helper flags
floating_model = (input_dtype == np.float32)

# ======================================================================
# Helper functions specific to detection interpretation and postprocessing
# ======================================================================

def iou(box_a, box_b):
    """
    Compute IoU between two boxes in absolute pixel coordinates.
    Boxes format: [x_min, y_min, x_max, y_max]
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def map_class_to_label(cls_id, labels_list):
    """
    Map class ID (float/int from TFLite output) to label string.
    Handles common off-by-one indexing differences by trying direct and -1 index.
    """
    cid = int(cls_id)
    if 0 <= cid < len(labels_list):
        return labels_list[cid]
    if 0 <= (cid - 1) < len(labels_list):
        return labels_list[cid - 1]
    return f"id {cid}"

def extract_detections_from_outputs(outputs, frame_w, frame_h, conf_thresh):
    """
    Phase 4.2 + 4.3 support: Interpret and post-process raw model outputs.

    outputs: list of numpy arrays from interpreter.get_tensor for each output_detail
    Returns a list of detections with fields:
      - bbox: [x_min, y_min, x_max, y_max] ints clipped to image
      - score: float
      - class_id: int
      - label: str (if available)
    """
    # Identify standard SSD outputs: boxes, classes, scores, num_detections
    boxes = None
    classes = None
    scores = None
    num_det = None

    # Assign by inspecting shapes and value ranges to be robust
    for arr in outputs:
        arr_np = np.array(arr)
        if arr_np.size == 1:
            num_det = int(np.squeeze(arr_np))
        elif arr_np.ndim >= 2 and arr_np.shape[-1] == 4:
            boxes = arr_np
        else:
            # scores vs classes: both [1, N]; scores are [0,1], classes likely > 1.0
            max_val = np.max(arr_np)
            min_val = np.min(arr_np)
            if max_val <= 1.0 and min_val >= 0.0:
                scores = arr_np
            else:
                classes = arr_np

    # Fallback handling in case some is None; typical TFLite SSD is [1,10,4], [1,10], [1,10], [1]
    if boxes is None or classes is None or scores is None:
        # Try another heuristic: order by output_details name if present
        # However, we stick to the guideline strictly; raise clear error if not found
        raise RuntimeError("Failed to interpret model outputs (boxes/classes/scores).")

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    # num_det may be absent in some models; infer from scores length
    if num_det is None:
        num_det = scores.shape[0] if scores.ndim == 1 else scores.shape[1] if scores.ndim == 2 else len(scores)

    detections = []
    for i in range(int(num_det)):
        score = float(scores[i])
        if score < conf_thresh:
            continue

        # TFLite SSD boxes are in normalized coordinates [ymin, xmin, ymax, xmax]
        y_min, x_min, y_max, x_max = boxes[i].tolist()

        x_min_abs = int(max(0, min(frame_w - 1, x_min * frame_w)))
        y_min_abs = int(max(0, min(frame_h - 1, y_min * frame_h)))
        x_max_abs = int(max(0, min(frame_w - 1, x_max * frame_w)))
        y_max_abs = int(max(0, min(frame_h - 1, y_max * frame_h)))

        # Ensure proper box ordering and non-zero area
        x1 = min(x_min_abs, x_max_abs)
        y1 = min(y_min_abs, y_max_abs)
        x2 = max(x_min_abs, x_max_abs)
        y2 = max(y_min_abs, y_max_abs)

        # Discard degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        cls_id = int(classes[i])
        label = map_class_to_label(cls_id, labels) if len(labels) > 0 else str(cls_id)

        detections.append({
            'bbox': [x1, y1, x2, y2],
            'score': score,
            'class_id': cls_id,
            'label': label
        })
    return detections

def compute_ap(scores, is_tp, total_positives):
    """
    Compute Average Precision (AP) using precision envelope integration.
    scores: list of floats
    is_tp: list of bools (True for TP, False for FP)
    total_positives: int (number of true positives in ground-truth sense for this proxy metric)
    """
    if total_positives <= 0 or len(scores) == 0:
        return 0.0

    # Sort by score descending
    order = np.argsort(-np.array(scores))
    tp_sorted = np.array(is_tp, dtype=np.int32)[order]
    fp_sorted = 1 - tp_sorted

    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)

    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / float(total_positives)

    # Make precision monotonically decreasing (precision envelope)
    # and integrate over recall changes
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Compute area under PR curve by summing Delta recall * precision
    ap = 0.0
    for i in range(1, mrec.size):
        delta_rec = mrec[i] - mrec[i - 1]
        ap += delta_rec * mpre[i]
    return float(ap)

def compute_proxy_map(eval_store):
    """
    Compute proxy mAP across classes from the running eval_store.
    eval_store structure:
      {
        class_id: {
            'scores': [float, ...], # Detection confidences (thresholded detections)
            'is_tp': [bool, ...],   # Temporal consistency TP flags vs previous frame
            'tp_total': int         # Total positives for proxy evaluation (sum of is_tp)
        }, ...
      }
    Returns mAP (float) across classes with tp_total > 0.
    """
    aps = []
    for cls_id, data in eval_store.items():
        tp_total = int(data.get('tp_total', 0))
        if tp_total <= 0:
            continue
        ap = compute_ap(data['scores'], data['is_tp'], tp_total)
        aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

def draw_detections_on_frame(frame, detections, map_value):
    """
    Draw bounding boxes and labels on the frame, and overlay current proxy mAP.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        label = det['label']

        # Box
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # Label text
        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, thickness=-1)
        cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # Overlay proxy mAP at top-left
    map_text = f"mAP: {map_value:.3f}"
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), thickness=2, lineType=cv2.LINE_AA)

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire Input Data: Read a single video file from the given input_path
if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0  # reasonable default if not present

# Prepare VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_width, orig_height))
if not writer.isOpened():
    raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

# Structures for proxy mAP evaluation across frames
# eval_store[class_id] = {'scores': [], 'is_tp': [], 'tp_total': 0}
eval_store = {}
prev_dets_by_class = {}  # Previous frame detections grouped by class_id: list of bboxes

frame_index = 0
t_start = time.time()

try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video

        frame_index += 1
        frame_h, frame_w = frame_bgr.shape[:2]

        # 2.2 Preprocess Data:
        # Convert BGR to RGB as most TFLite SSD models expect RGB input
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (input_width, input_height))
        input_data = np.expand_dims(resized, axis=0)

        # 2.3 Quantization Handling
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # Ensure dtype matches model input type
            input_data = np.asarray(input_data, dtype=input_dtype)

        # =========================
        # Phase 3: Inference
        # =========================
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # =========================
        # Phase 4: Output Interpretation & Handling
        # =========================

        # 4.1 Get Output Tensor(s)
        raw_outputs = [interpreter.get_tensor(od['index']) for od in output_details]

        # 4.2 Interpret Results
        detections = extract_detections_from_outputs(raw_outputs, frame_w, frame_h, CONF_THRESHOLD)

        # 4.3 Post-processing:
        # - Confidence thresholding and coordinate scaling already applied in extract_detections_from_outputs
        # - Perform simple greedy matching against previous frame by class to build proxy TP/FP for mAP

        # Group current detections by class
        current_by_class = {}
        for det in detections:
            cid = det['class_id']
            current_by_class.setdefault(cid, []).append(det)

        # Initialize eval_store entries
        for cid in current_by_class.keys():
            if cid not in eval_store:
                eval_store[cid] = {'scores': [], 'is_tp': [], 'tp_total': 0}
        for cid in prev_dets_by_class.keys():
            if cid not in eval_store:
                eval_store[cid] = {'scores': [], 'is_tp': [], 'tp_total': 0}

        # For each class, match current detections to previous detections via IoU >= 0.5
        new_prev_dets_by_class = {}
        for cid, curr_list in current_by_class.items():
            prev_list = prev_dets_by_class.get(cid, [])
            prev_used = [False] * len(prev_list)

            # For future prev tracking for next frame
            new_prev_dets_by_class[cid] = [det['bbox'] for det in curr_list]

            for det in curr_list:
                bbox = det['bbox']
                score = det['score']

                matched = False
                # Greedy match: find best IoU with unused previous boxes
                best_iou = 0.0
                best_j = -1
                for j, prev_bbox in enumerate(prev_list):
                    if prev_used[j]:
                        continue
                    ov = iou(bbox, prev_bbox)
                    if ov >= 0.5 and ov > best_iou:
                        best_iou = ov
                        best_j = j

                if best_j >= 0:
                    prev_used[best_j] = True
                    matched = True

                # Update evaluation store
                eval_store[cid]['scores'].append(score)
                eval_store[cid]['is_tp'].append(bool(matched))
                # Count positives as matched pairs (proxy positives)
                if matched:
                    eval_store[cid]['tp_total'] += 1

        # Also carry forward classes that had only previous detections but none current
        for cid, prev_list in prev_dets_by_class.items():
            if cid not in new_prev_dets_by_class:
                new_prev_dets_by_class[cid] = []

        # Update prev for next iteration
        prev_dets_by_class = new_prev_dets_by_class

        # Compute running proxy mAP
        running_map = compute_proxy_map(eval_store)

        # 4.4 Handle Output: draw and write frame
        output_frame = frame_bgr.copy()
        draw_detections_on_frame(output_frame, detections, running_map)
        writer.write(output_frame)

        # 4.5 Loop Continuation: continue until video ends; no external break

finally:
    # =========================
    # Phase 5: Cleanup
    # =========================
    cap.release()
    writer.release()
    t_elapsed = time.time() - t_start

# Print summary and final mAP
final_map = compute_proxy_map(eval_store)
print("Processing finished.")
print(f"Input video: {INPUT_PATH}")
print(f"Output video: {OUTPUT_PATH}")
print(f"Frames processed: {frame_index}")
print(f"Elapsed time: {t_elapsed:.2f} s, FPS (effective): {frame_index / max(t_elapsed, 1e-9):.2f}")
print(f"Calculated mAP (proxy, temporal consistency): {final_map:.4f}")