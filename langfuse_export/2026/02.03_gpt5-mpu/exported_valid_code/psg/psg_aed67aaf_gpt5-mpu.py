#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single input video file using a TFLite SSD MobileNet v1 model.
It writes an output video with bounding boxes and labels overlaid, and overlays a running "mAP" estimate
(computed without ground-truth as a self-consistency proxy; see comments in Phase 4.2 for details).

Phases implemented according to the provided Programming Guidelines:
- Phase 1: Setup
- Phase 2: Input Acquisition & Preprocessing Loop (explicit)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling Loop, including:
  - 4.2: Interpret Results (explicit)
  - 4.3: Post-processing (explicit)
- Phase 5: Cleanup

Dependencies:
- Standard libraries: os, time, numpy, cv2 (video/image processing)
- TFLite runtime: ai_edge_litert.interpreter.Interpreter
"""

import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Phase 1: Setup
# =========================

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5  # For drawing/display

# 1.3 Load Labels (if provided and relevant)
def load_labels(path):
    labels = []
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract primary input tensor info
in_idx = input_details[0]['index']
in_shape = input_details[0]['shape']
# Expect shape [1, height, width, 3]
if len(in_shape) != 4 or in_shape[0] != 1 or in_shape[-1] != 3:
    raise ValueError("Unexpected model input shape: {}".format(in_shape))
in_height, in_width = int(in_shape[1]), int(in_shape[2])
in_dtype = input_details[0]['dtype']
floating_model = (in_dtype == np.float32)

# Helper to get consistent color per class id
def class_color(cid):
    # Generate a color based on the class id for consistency.
    np.random.seed(cid + 12345)
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    # OpenCV uses BGR
    return (color[0], color[1], color[2])

# Directory prep for output
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# Input Method: Read a single video file from the given input_path
# =========================

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Failed to open input video: {}".format(input_path))

# Read input video properties
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or np.isnan(fps) or fps <= 0:
    fps = 25.0  # Fallback if FPS is unavailable
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError("Failed to open output video for writing: {}".format(output_path))

# Structures for running "mAP" proxy computation (no ground-truth available)
# We compute a self-consistency proxy AP per class:
# - For each frame and class, the highest-scoring detection is treated as TP candidate (1 per frame),
#   and all other detections for that class in that frame are treated as FP.
# - Aggregate across frames and compute AP as area under precision-recall curve over scores.
per_class_scores_flags = {}  # dict: class_id -> list of tuples (score, is_tp)
frames_processed = 0

def compute_ap_from_scores_flags(sf_list):
    """
    Compute AP from list of (score, is_tp) entries for a single class, across frames processed so far.
    Uses a standard precision-recall area with monotonically decreasing precision envelope.
    If there are no positives (no frames with detections), returns None to indicate "skip in mAP".
    """
    if not sf_list:
        return None

    # Count positives (is_tp True)
    num_positives = sum(1 for _, is_tp in sf_list if is_tp)
    if num_positives == 0:
        return None

    # Sort by score descending
    sf_sorted = sorted(sf_list, key=lambda x: x[0], reverse=True)

    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for score, is_tp in sf_sorted:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / num_positives
        precisions.append(precision)
        recalls.append(recall)

    # Compute AP via precision envelope integration (VOC 2010 style)
    precisions = np.array(precisions, dtype=np.float32)
    recalls = np.array(recalls, dtype=np.float32)

    # Monotonic precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        if precisions[i] < precisions[i + 1]:
            precisions[i] = precisions[i + 1]

    # Identify points where recall changes
    # We need to include the starting (0, p0) and ending points; handle accordingly
    unique_recalls, indices = np.unique(recalls, return_index=True)
    # Sort indices to have increasing recall
    order = np.argsort(unique_recalls)
    unique_recalls = unique_recalls[order]
    indices = indices[order]

    # For integration, we can approximate as sum over recall diffs with corresponding precision
    ap = 0.0
    prev_recall = 0.0
    for r, idx in zip(unique_recalls, indices):
        ap += precisions[idx] * (r - prev_recall)
        prev_recall = r

    return float(ap)

def compute_running_map(per_class_dict):
    aps = []
    for cid, sf_list in per_class_dict.items():
        ap = compute_ap_from_scores_flags(sf_list)
        if ap is not None:
            aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frames_processed += 1

    # 2.2 Preprocess data to match model input
    # Convert BGR to RGB, resize to model input size, add batch dimension
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_width, in_height), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)

    # Convert dtype and normalize if floating model
    if floating_model:
        # Normalize to [-1, 1] as typical for SSD MobileNet v1
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Ensure dtype matches (usually uint8)
        input_data = input_data.astype(in_dtype, copy=False)

    # =========================
    # Phase 3: Inference
    # =========================
    interpreter.set_tensor(in_idx, input_data)
    interpreter.invoke()

    # =========================
    # Phase 4: Output Interpretation & Handling Loop
    # =========================

    # 4.1 Get Output Tensors
    # Typical SSD TFLite outputs:
    # - boxes: [1, N, 4] (ymin, xmin, ymax, xmax) normalized to 0..1
    # - classes: [1, N] (float indices)
    # - scores: [1, N] (0..1)
    # - num_detections: [1]
    boxes = None
    classes = None
    scores = None
    num_detections = None

    for od in output_details:
        out = interpreter.get_tensor(od['index'])
        # Identify tensors by shape and value ranges
        if isinstance(out, np.ndarray):
            if out.ndim == 3 and out.shape[0] == 1 and out.shape[2] == 4:
                boxes = out[0]
            elif out.ndim == 2 and out.shape[0] == 1:
                arr = out[0]
                if arr.dtype.kind == 'f' and np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
                    # Likely scores
                    scores = arr
                else:
                    # Likely classes (float indices)
                    classes = arr
            elif out.size == 1:
                # num_detections
                num_detections = int(np.round(float(out.flatten()[0])))

    # Fallbacks in case num_detections not provided
    max_detections = None
    if boxes is not None:
        max_detections = boxes.shape[0]
    if scores is not None:
        max_detections = scores.shape[0] if max_detections is None else min(max_detections, scores.shape[0])
    if classes is not None:
        max_detections = classes.shape[0] if max_detections is None else min(max_detections, classes.shape[0])
    if num_detections is None:
        num_detections = max_detections if max_detections is not None else 0
    num_detections = int(num_detections)

    # Defensive checks
    if boxes is None or scores is None or classes is None:
        # If model outputs aren't recognized, skip frame safely
        cv2.putText(frame, "Model output parsing error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(frame)
        continue

    # Ensure shapes align
    num_detections = min(num_detections, boxes.shape[0], scores.shape[0], classes.shape[0])

    # 4.2 Interpret Results
    # - Map class indices to human-readable labels if available.
    # - Prepare list of detections for drawing and for mAP proxy accumulation.
    dets_for_frame = []  # list of dicts: {'cid': int, 'score': float, 'box': (x1,y1,x2,y2), 'label': str}

    for i in range(num_detections):
        score = float(scores[i])
        # Class id: handle both 0-based and 1-based indexing defensively
        raw_cid = int(classes[i])
        if raw_cid < 0:
            continue
        cid = raw_cid
        if labels and (cid >= len(labels)) and (cid - 1) >= 0 and (cid - 1) < len(labels):
            # If class id looks 1-based, shift to 0-based for label lookup
            cid = cid - 1

        label_name = None
        if labels and (0 <= cid < len(labels)):
            label_name = labels[cid]
        else:
            label_name = f"id_{raw_cid}"

        # 4.3 Post-processing: scale and clip boxes to frame boundaries
        # boxes[i] expected [ymin, xmin, ymax, xmax] normalized to 0..1
        ymin, xmin, ymax, xmax = boxes[i]
        # Clip normalized range
        ymin = float(np.clip(ymin, 0.0, 1.0))
        xmin = float(np.clip(xmin, 0.0, 1.0))
        ymax = float(np.clip(ymax, 0.0, 1.0))
        xmax = float(np.clip(xmax, 0.0, 1.0))
        # Scale to frame size
        x1 = int(xmin * frame_w)
        y1 = int(ymin * frame_h)
        x2 = int(xmax * frame_w)
        y2 = int(ymax * frame_h)
        # Clip to frame bounds
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y2 = max(0, min(y2, frame_h - 1))

        # Store detection entry
        dets_for_frame.append({
            'cid': cid,
            'raw_cid': raw_cid,
            'score': score,
            'box': (x1, y1, x2, y2),
            'label': label_name
        })

    # Aggregate detections for mAP proxy computation:
    # For each class in current frame, mark the highest-score detection as TP; others as FP.
    if dets_for_frame:
        # Group by class
        by_class = {}
        for d in dets_for_frame:
            by_class.setdefault(d['cid'], []).append(d)
        for cid, det_list in by_class.items():
            # Determine the top-scoring detection
            top_idx = int(np.argmax([d['score'] for d in det_list]))
            for idx, d in enumerate(det_list):
                is_tp = (idx == top_idx)
                per_class_scores_flags.setdefault(cid, []).append((d['score'], is_tp))

    # Compute running mAP (proxy) across processed frames
    running_map = compute_running_map(per_class_scores_flags)

    # 4.4 Handle Output: Draw detections and overlay mAP
    # Draw thresholded detections
    for d in dets_for_frame:
        if d['score'] < CONF_THRESHOLD:
            continue  # Only draw those above threshold
        x1, y1, x2, y2 = d['box']
        color = class_color(d['cid'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{d['label']}: {d['score']:.2f}"
        # Put label background for readability
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th_box = th + baseline + 4
        cv2.rectangle(frame, (x1, max(0, y1 - th_box)), (x1 + tw + 4, y1), color, thickness=-1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Overlay running mAP (proxy) at top-left
    map_text = f"mAP (proxy): {running_map * 100:.2f}%"
    (tw, th), baseline = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (8 + tw + 8, 8 + th + baseline + 8), (0, 0, 0), thickness=-1)
    cv2.putText(frame, map_text, (12, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Write frame to output video
    writer.write(frame)

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
writer.release()

# Print final mAP proxy
final_map = compute_running_map(per_class_scores_flags)
print("Processing completed.")
print(f"Frames processed: {frames_processed}")
print(f"Final mAP (proxy, self-consistency without ground-truth): {final_map:.4f} ({final_map*100:.2f}%)")
print(f"Output saved to: {output_path}")