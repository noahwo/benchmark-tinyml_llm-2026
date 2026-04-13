#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single input video using a TFLite SSD model.
It writes an output video with bounding boxes, labels, and a running temporal mAP (self-supervised)
estimate displayed on each frame.

Phases implemented according to the provided Programming Guideline:
- Phase 1: Setup
- Phase 2: Input Acquisition & Preprocessing Loop
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling Loop (including 4.2 interpretation and 4.3 post-processing)
- Phase 5: Cleanup

Notes:
- Temporal mAP is computed self-supervised by treating ground truths as detections from the previous frame,
  matched to current detections by IoU >= 0.5 and same class. This provides a temporal consistency metric
  without requiring external annotations.
"""

import os
import sys
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: Import interpreter literally as specified
from ai_edge_litert.interpreter import Interpreter

def load_labels(label_path):
    """
    Load labels from a text file (one label per line).
    Returns:
        labels (list of str)
        label_offset (int): 1 if first label is '???' or 'background', else 0
    """
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name == '':
                continue
            labels.append(name)
    label_offset = 1 if (len(labels) > 0 and labels[0].strip().lower() in ('???', 'background')) else 0
    return labels, label_offset

def ensure_dir(path):
    """Ensure directory exists for the given path."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def letterbox_resize(image, target_w, target_h):
    """
    Optionally could letterbox; however many SSD models expect simple resize.
    For simplicity and performance, use direct resize keeping aspect ratio ignored,
    as typical SSD Mobilenet was trained on fixed size inputs.
    """
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def preprocess_frame_bgr_to_model_input(frame_bgr, input_details):
    """
    Convert BGR frame to the model input tensor based on input_details.
    - Convert to RGB
    - Resize to expected input spatial dimensions
    - Handle dtype conversion and normalization for floating models
    """
    # Determine input shape: [1, height, width, channels]
    in_shape = input_details[0]['shape']
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected model input shape: {in_shape}")
    in_h, in_w = int(in_shape[1]), int(in_shape[2])

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = letterbox_resize(frame_rgb, in_w, in_h)

    # Prepare input tensor
    input_dtype = input_details[0]['dtype']
    input_data = resized

    # Phase 2.3: Quantization Handling
    floating_model = (input_dtype == np.float32)
    if floating_model:
        # Normalize to [-1, 1] as per guideline
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # For quantized models, cast to the required dtype
        input_data = np.asarray(input_data, dtype=input_dtype)

    # Expand dims to [1,H,W,C]
    input_data = np.expand_dims(input_data, axis=0)
    return input_data, (in_w, in_h), floating_model

def iou_xyxy(boxA, boxB):
    """
    Compute IoU between two boxes in absolute pixel coordinates: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = (areaA + areaB - inter_area)
    if denom <= 0.0:
        return 0.0
    return inter_area / denom

def compute_ap(precisions, recalls):
    """
    Compute Average Precision using the standard all-point interpolation method:
    1) Append boundary points
    2) Make precision monotonically non-increasing
    3) Sum over recall steps of delta_recall * precision_envelope

    Args:
        precisions: list or np.array of precision values sorted by descending score
        recalls: list or np.array of recall values sorted by descending score

    Returns:
        AP (float)
    """
    if len(precisions) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area under PR curve
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)

def compute_map(per_class_scores, per_class_tps, per_class_gt_count):
    """
    Compute mean Average Precision across classes present (gt_count > 0).
    The TP/FP flags should correspond to predictions sorted by score DESC per class at evaluation time.
    Here, we re-sort within this function to ensure correct ordering across frames.

    Args:
        per_class_scores: dict[class_id] -> list of scores (float)
        per_class_tps: dict[class_id] -> list of 0/1 flags
        per_class_gt_count: dict[class_id] -> int count of ground-truth boxes

    Returns:
        mAP (float)
    """
    aps = []
    classes_with_gt = [cid for cid, gtc in per_class_gt_count.items() if gtc > 0]
    if len(classes_with_gt) == 0:
        return 0.0

    for cid in classes_with_gt:
        scores = np.asarray(per_class_scores.get(cid, []), dtype=np.float32)
        tps = np.asarray(per_class_tps.get(cid, []), dtype=np.int32)
        gtc = int(per_class_gt_count.get(cid, 0))
        if gtc <= 0 or scores.size == 0:
            aps.append(0.0)
            continue

        # Sort by score descending
        order = np.argsort(-scores)
        tps_sorted = tps[order]
        fps_sorted = 1 - tps_sorted

        cum_tp = np.cumsum(tps_sorted)
        cum_fp = np.cumsum(fps_sorted)

        # Avoid division by zero
        denom = (cum_tp + cum_fp)
        denom[denom == 0] = 1e-9
        precisions = cum_tp / denom
        # Recall relative to ground truth count for this class
        recalls = cum_tp / max(1, gtc)

        ap = compute_ap(precisions, recalls)
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

def color_for_class(cid):
    """
    Deterministically compute a color (B,G,R) for a given class id.
    """
    # Simple hash to color mapping
    r = (37 * (cid + 1)) % 255
    g = (17 * (cid + 1)) % 255
    b = (29 * (cid + 1)) % 255
    return int(b), int(g), int(r)

def parse_tflite_outputs(interpreter, output_details):
    """
    Retrieve and parse TFLite SSD outputs into standardized arrays.
    Returns:
        boxes: np.ndarray [N,4] (ymin, xmin, ymax, xmax), usually normalized [0,1]
        classes: np.ndarray [N] of class indices (float or int)
        scores: np.ndarray [N] of confidence scores [0,1]
        num: int, number of valid detections (if provided by model), else len(scores)
    """
    # Phase 4.1: Get Output Tensors
    outputs = []
    for od in output_details:
        outputs.append(interpreter.get_tensor(od['index']))
    # Flatten output list
    # Typical: 4 tensors: boxes [1,N,4], classes [1,N], scores [1,N], num [1]
    boxes = None
    classes = None
    scores = None
    num = None

    # First detect boxes by looking for tensor with last dim == 4
    for arr in outputs:
        if arr.ndim >= 2 and arr.shape[-1] == 4:
            boxes = arr
            break
    # Detect classes and scores by range and shape
    tmp_others = [arr for arr in outputs if arr is not boxes]
    for arr in tmp_others:
        arr_squeezed = np.squeeze(arr)
        if arr_squeezed.ndim == 0:
            # num_detections likely
            num = int(arr_squeezed)
        elif arr.dtype in (np.float32, np.float16):
            # Scores likely in [0,1]
            maxv = float(np.max(arr_squeezed)) if arr_squeezed.size > 0 else 0.0
            minv = float(np.min(arr_squeezed)) if arr_squeezed.size > 0 else 0.0
            if 0.0 <= minv and maxv <= 1.0:
                scores = arr
            else:
                classes = arr
        elif np.issubdtype(arr.dtype, np.integer):
            # Could be classes (int) or num
            if arr.size == 1:
                num = int(np.squeeze(arr))
            else:
                classes = arr

    # Fall-back logic if any is missing
    if boxes is None:
        # Try to find something that looks like boxes
        for arr in outputs:
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr
                break
    if classes is None:
        # Try to find non-score arrays leftover
        for arr in outputs:
            if arr is boxes:
                continue
            arr_squeezed = np.squeeze(arr)
            if arr_squeezed.ndim == 1 and arr_squeezed.size > 1:
                # decide by range if it's classes or scores
                maxv = float(np.max(arr_squeezed))
                minv = float(np.min(arr_squeezed))
                if not (0.0 <= minv and maxv <= 1.0):
                    classes = arr
                    break
    if scores is None:
        for arr in outputs:
            if arr is boxes or arr is classes:
                continue
            arr_squeezed = np.squeeze(arr)
            if arr_squeezed.ndim == 1 and arr_squeezed.size > 1:
                maxv = float(np.max(arr_squeezed))
                minv = float(np.min(arr_squeezed))
                if 0.0 <= minv and maxv <= 1.0:
                    scores = arr
                    break
    if num is None:
        # If num not provided, infer N from classes or scores
        if classes is not None:
            num = int(np.squeeze(classes).shape[0])
        elif scores is not None:
            num = int(np.squeeze(scores).shape[0])
        elif boxes is not None:
            b = np.squeeze(boxes)
            if b.ndim == 2:
                num = int(b.shape[0])
            else:
                num = 0
        else:
            num = 0

    # Squeeze batch dimension if present
    if boxes is not None and boxes.ndim >= 3:
        boxes = boxes[0]
    elif boxes is not None and boxes.ndim == 2:
        pass
    if classes is not None and classes.ndim >= 2:
        classes = classes[0]
    if scores is not None and scores.ndim >= 2:
        scores = scores[0]

    # Slice to num if shapes are longer
    if boxes is not None and boxes.shape[0] >= num:
        boxes = boxes[:num]
    if classes is not None and classes.shape[0] >= num:
        classes = classes[:num]
    if scores is not None and scores.shape[0] >= num:
        scores = scores[:num]

    # Ensure numpy arrays or set defaults
    boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 4), dtype=np.float32)
    classes = np.asarray(classes) if classes is not None else np.zeros((0,), dtype=np.float32)
    scores = np.asarray(scores) if scores is not None else np.zeros((0,), dtype=np.float32)

    return boxes, classes, scores, num

def draw_detections(frame, detections, labels, label_offset):
    """
    Draw bounding boxes and labels on the frame.
    detections: list of dicts: {'bbox':[x1,y1,x2,y2], 'score':float, 'class_id':int}
    """
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cid = int(det['class_id'])
        score = float(det['score'])
        color = color_for_class(cid)
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        # Label text
        label_text = None
        # Map class id to label considering offset if present
        idx = cid + label_offset
        if 0 <= idx < len(labels):
            label_text = labels[idx]
        else:
            label_text = f'class_{cid}'
        text = f"{label_text} {score:.2f}"
        # Put text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, thickness=-1)
        cv2.putText(frame, text, (x1, max(0, y1 - baseline)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def clip_box_to_image(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Clip bounding box coordinates to image boundaries and convert to int pixel coordinates.
    """
    x1 = int(np.clip(xmin, 0, img_w - 1))
    y1 = int(np.clip(ymin, 0, img_h - 1))
    x2 = int(np.clip(xmax, 0, img_w - 1))
    y2 = int(np.clip(ymax, 0, img_h - 1))
    # Ensure x1<=x2, y1<=y2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def main():
    # 1.2 Paths/Parameters
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')

    # 1.3 Load Labels (Conditional)
    labels, label_offset = load_labels(label_path)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) < 1:
        print("Error: Model has no input tensors.", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: Read a single video file from the given input_path
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}", file=sys.stderr)
        sys.exit(1)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}", file=sys.stderr)
        sys.exit(1)
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps is None or in_fps <= 1e-3 or np.isnan(in_fps):
        in_fps = 25.0  # Fallback FPS
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output writer
    ensure_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, in_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: Could not open VideoWriter for {output_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Temporal mAP accumulators
    iou_threshold = 0.5
    per_class_scores = {}     # class_id -> list[score]
    per_class_tps = {}        # class_id -> list[int]
    per_class_gt_count = {}   # class_id -> int
    prev_gts = []             # list of dicts: {'bbox':[x1,y1,x2,y2], 'class_id':int}

    frame_count = 0
    t0 = time.time()
    final_map = 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # 2.2 Preprocess Data based on model input details
        input_data, (in_w, in_h), floating_model = preprocess_frame_bgr_to_model_input(frame_bgr, input_details)

        # Phase 3: Inference
        # 3.1 Set Input Tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling Loop
        # 4.1 Get Output Tensor(s)
        boxes_raw, classes_raw, scores_raw, num_raw = parse_tflite_outputs(interpreter, output_details)

        # 4.2 Interpret Results
        # Convert boxes to pixel coordinates, apply confidence threshold and clipping
        H, W = frame_bgr.shape[:2]
        detections = []
        # Determine if boxes are normalized ([0,1]) by checking value range
        boxes_are_normalized = False
        if boxes_raw.size > 0:
            max_box_val = float(np.max(boxes_raw))
            boxes_are_normalized = max_box_val <= 1.5  # heuristic

        for i in range(boxes_raw.shape[0]):
            score = float(scores_raw[i]) if i < scores_raw.shape[0] else 0.0
            if score < confidence_threshold:
                continue
            # Class id as int
            cls_val = classes_raw[i] if i < classes_raw.shape[0] else 0
            try:
                class_id = int(cls_val)
            except Exception:
                class_id = int(float(cls_val))
            # Box format from TFLite SSD: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes_raw[i].tolist()
            if boxes_are_normalized:
                x1 = int(xmin * W)
                y1 = int(ymin * H)
                x2 = int(xmax * W)
                y2 = int(ymax * H)
            else:
                x1 = int(xmin)
                y1 = int(ymin)
                x2 = int(xmax)
                y2 = int(ymax)
            # 4.3 Post-processing: Bounding box clipping
            x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, W, H)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': score,
                'class_id': class_id
            })

        # 4.2 Use labels to map indices to names when drawing
        draw_detections(frame_bgr, detections, labels, label_offset)

        # Temporal mAP calculation (self-supervised)
        # Create GTs from previous frame detections (above threshold already)
        # Evaluate current predictions against prev_gts using IoU >= 0.5 and same class
        if len(prev_gts) > 0:
            # Count GTs per class
            gt_count_frame = {}
            for gt in prev_gts:
                cid = int(gt['class_id'])
                gt_count_frame[cid] = gt_count_frame.get(cid, 0) + 1
            # Update global GT count
            for cid, cnt in gt_count_frame.items():
                per_class_gt_count[cid] = per_class_gt_count.get(cid, 0) + cnt

            # Prepare matching
            matched_gt_indices_by_class = {}
            for cid in gt_count_frame.keys():
                matched_gt_indices_by_class[cid] = set()

            # Sort predictions by score descending for fair evaluation
            preds_sorted_idx = sorted(range(len(detections)), key=lambda k: -detections[k]['score'])

            for pi in preds_sorted_idx:
                pred = detections[pi]
                pcid = int(pred['class_id'])
                pbox = pred['bbox']
                pscore = float(pred['score'])

                # Find best GT match of same class with IoU >= threshold that is not matched yet
                best_iou = 0.0
                best_gt_idx = -1
                for gti, gt in enumerate(prev_gts):
                    gcid = int(gt['class_id'])
                    if gcid != pcid:
                        continue
                    if gti in matched_gt_indices_by_class[pcid]:
                        continue
                    iou = iou_xyxy(pbox, gt['bbox'])
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gti

                # Record TP/FP per-class
                if pcid not in per_class_scores:
                    per_class_scores[pcid] = []
                    per_class_tps[pcid] = []
                per_class_scores[pcid].append(pscore)
                if best_gt_idx >= 0:
                    per_class_tps[pcid].append(1)
                    matched_gt_indices_by_class[pcid].add(best_gt_idx)
                else:
                    per_class_tps[pcid].append(0)

            # Compute running mAP
            final_map = compute_map(per_class_scores, per_class_tps, per_class_gt_count)
        else:
            # No GT for the first frame; mAP stays 0.0
            final_map = 0.0

        # Overlay mAP on the frame
        map_text = f"mAP@IoU=0.50 (temporal): {final_map:.3f}"
        cv2.putText(frame_bgr, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (5, 245, 5), 2, cv2.LINE_AA)

        # 4.4 Handle Output: write frame to output video
        writer.write(frame_bgr)

        # 4.5 Loop continuation: prepare GTs for next frame
        prev_gts = [{'bbox': det['bbox'], 'class_id': det['class_id']} for det in detections]

    # Phase 5: Cleanup
    cap.release()
    writer.release()
    t1 = time.time()

    # Summary output to console
    print(f"Processed {frame_count} frame(s) in {t1 - t0:.2f} s. Average FPS: {frame_count / max(1e-6, (t1 - t0)):.2f}")
    print(f"Output saved to: {output_path}")
    print(f"Final temporal mAP@0.50: {final_map:.3f}")

if __name__ == "__main__":
    main()