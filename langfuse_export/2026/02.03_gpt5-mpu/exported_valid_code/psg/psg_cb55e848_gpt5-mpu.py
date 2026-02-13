#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single input video using a TFLite SSD model.
It annotates detected objects with bounding boxes and labels, and writes the annotated
video to the specified output path. Additionally, it computes and overlays a running
mAP (mean Average Precision) estimate based on self-consistency (per-class NMS clusters
are treated as pseudo ground truths and duplicates are false positives).

Phases implemented as per guideline:
- Phase 1: Setup (imports, paths, labels, interpreter, model details)
- Phase 2: Input Acquisition & Preprocessing Loop (Read frames from a video file)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling (detection processing, thresholding, scaling, clipping, drawing, mAP)
- Phase 5: Cleanup
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Import Interpreter exactly as specified
from ai_edge_litert.interpreter import Interpreter


def load_labels(label_file_path):
    """
    Load labels from a text file. Each line corresponds to a label.
    Empty lines and comment lines are ignored.
    """
    labels = []
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith('#'):
                    continue
                labels.append(s)
    except Exception as ex:
        print(f"[WARN] Could not read label file '{label_file_path}': {ex}")
        labels = []
    return labels


def preprocess_frame_bgr_to_model_input(frame_bgr, input_shape, floating_model):
    """
    Convert a BGR frame to the model's expected input tensor.
    - input_shape: [1, height, width, channels]
    - floating_model: True if model input dtype is float32
    Returns: input_data numpy array with shape matching input_shape
    """
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    # Convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    # Add batch dimension
    input_data = np.expand_dims(resized, axis=0)
    if floating_model:
        # Normalize to [-1, 1] as per many MobileNet-based models
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Keep uint8 [0,255]
        input_data = np.uint8(input_data)
    return input_data


def clip_box(xmin, ymin, xmax, ymax, w, h):
    """
    Clip the bounding box coordinates to valid image bounds.
    """
    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w - 1))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h - 1))
    return xmin, ymin, xmax, ymax


def iou_xyxy(boxA, boxB):
    """
    Compute IoU between two boxes in [xmin, ymin, xmax, ymax] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = areaA + areaB - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def per_class_pseudo_gt_and_tp_flags(dets_for_class, iou_thresh=0.5):
    """
    Given detections for a class for a single frame (list of (score, [xmin,ymin,xmax,ymax])),
    perform a greedy NMS-like clustering:
    - The highest-confidence box in a cluster is treated as a true positive (TP).
    - Any other box in that cluster (IoU >= iou_thresh to the cluster representative) is a false positive (FP).
    Returns:
    - det_records: list of (score, is_tp) for each detection in dets_for_class,
                   ordered by descending score (consistent for AP calculation).
    - num_clusters: number of clusters (pseudo ground truths) in this frame for this class.
    """
    if not dets_for_class:
        return [], 0

    # Sort by descending score
    sorted_indices = sorted(range(len(dets_for_class)), key=lambda i: dets_for_class[i][0], reverse=True)
    assigned = [False] * len(dets_for_class)
    det_records = []
    num_clusters = 0

    for idx in sorted_indices:
        if assigned[idx]:
            continue
        # Start a new cluster with this detection as representative (TP)
        num_clusters += 1
        assigned[idx] = True
        det_records.append((dets_for_class[idx][0], True))  # TP

        # Suppress duplicates (FPs) for the cluster
        rep_box = dets_for_class[idx][1]
        for jdx in sorted_indices:
            if assigned[jdx]:
                continue
            iou_val = iou_xyxy(rep_box, dets_for_class[jdx][1])
            if iou_val >= iou_thresh:
                assigned[jdx] = True
                det_records.append((dets_for_class[jdx][0], False))  # FP

    # det_records is in the order of cluster creation and suppression (descending score across clusters)
    # Already suitable for global AP aggregation when concatenated across frames.
    return det_records, num_clusters


def compute_ap_from_records(det_records, total_gt):
    """
    Compute Average Precision (AP) given detection records for a class across frames
    and total pseudo ground truth count (sum of clusters) for that class.

    det_records: list of (score, is_tp) across frames
    total_gt: int, number of pseudo-gt instances for the class

    Uses the precision envelope integral method.
    """
    if total_gt <= 0 or len(det_records) == 0:
        return 0.0

    # Sort by confidence descending across all records
    det_records_sorted = sorted(det_records, key=lambda x: x[0], reverse=True)
    tps = []
    fps = []
    for _, is_tp in det_records_sorted:
        tps.append(1 if is_tp else 0)
        fps.append(0 if is_tp else 1)

    tps = np.array(tps, dtype=np.float32)
    fps = np.array(fps, dtype=np.float32)

    cum_tps = np.cumsum(tps)
    cum_fps = np.cumsum(fps)

    recalls = cum_tps / float(total_gt)
    precisions = cum_tps / np.maximum(cum_tps + cum_fps, 1e-12)

    # Precision envelope
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integration points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def compute_running_map(per_class_records, per_class_total_gt):
    """
    Compute mean AP across classes with non-zero pseudo ground truths.
    per_class_records: dict[class_id] -> list of (score, is_tp)
    per_class_total_gt: dict[class_id] -> int
    Returns mAP float in [0,1].
    """
    aps = []
    for cid, total_gt in per_class_total_gt.items():
        if total_gt <= 0:
            continue
        records = per_class_records.get(cid, [])
        ap = compute_ap_from_records(records, total_gt)
        aps.append(ap)
    if not aps:
        return 0.0
    return float(np.mean(aps))


def main():
    # 1.2 Paths/Parameters
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')  # Provided as string; convert to float

    # 1.3 Load Labels (if provided and relevant)
    labels = load_labels(label_path)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine input tensor properties
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # Expected [1, h, w, c]
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # Attempt to map output tensors
    # Many SSD TFLite models produce 4 outputs: boxes, classes, scores, num_detections
    # We'll assume the common order; if not, we try to infer by shapes.
    boxes_idx = None
    classes_idx = None
    scores_idx = None
    num_idx = None

    if len(output_details) >= 4:
        # Try common order
        boxes_idx = output_details[0]['index']
        classes_idx = output_details[1]['index']
        scores_idx = output_details[2]['index']
        num_idx = output_details[3]['index']
        # Basic sanity checks; if not matching expected shapes, attempt inference by shape
        try:
            boxes_shape = output_details[0]['shape']
            classes_shape = output_details[1]['shape']
            scores_shape = output_details[2]['shape']
            num_shape = output_details[3]['shape']
            if not (len(boxes_shape) == 3 and boxes_shape[-1] == 4):
                raise ValueError("Boxes shape not as expected")
            if not (len(num_shape) == 1 and num_shape[0] == 1):
                raise ValueError("Num detections shape not as expected")
        except Exception:
            # Infer by shapes
            boxes_idx = classes_idx = scores_idx = num_idx = None
            for od in output_details:
                shape = od['shape']
                if len(shape) == 3 and shape[-1] == 4:
                    boxes_idx = od['index']
                elif len(shape) == 1 and shape[0] == 1:
                    num_idx = od['index']
                else:
                    # classes and scores typically [1, N]
                    # Prefer to assign scores to float tensors and classes to non-float or also float
                    if scores_idx is None and od['dtype'] == np.float32:
                        scores_idx = od['index']
                    elif classes_idx is None:
                        classes_idx = od['index']
            # If still ambiguous, default to common order
            if None in (boxes_idx, classes_idx, scores_idx, num_idx):
                boxes_idx = output_details[0]['index']
                classes_idx = output_details[1]['index']
                scores_idx = output_details[2]['index']
                num_idx = output_details[3]['index']
    else:
        raise RuntimeError("Unexpected number of output tensors from model.")

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: Read a single video file from the given input_path
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    # Prepare output writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback if video doesn't report FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    # Data structures for running mAP computation
    per_class_records = {}     # class_id -> list of (score, is_tp)
    per_class_total_gt = {}    # class_id -> int
    frame_count = 0
    inference_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            frame_count += 1

            # 2.2 Preprocess Data
            input_data = preprocess_frame_bgr_to_model_input(frame, input_shape, floating_model)

            # 2.3 Quantization Handling for floating model already handled; uint8 kept as is.

            # Phase 3: Inference
            # 3.1 Set Input Tensor
            interpreter.set_tensor(input_index, input_data)
            # 3.2 Run Inference
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            inference_times.append((t1 - t0) * 1000.0)  # ms

            # Phase 4: Output Interpretation & Handling
            # 4.1 Get Output Tensors
            boxes = interpreter.get_tensor(boxes_idx)     # Expected shape [1, N, 4]
            classes = interpreter.get_tensor(classes_idx) # Expected shape [1, N]
            scores = interpreter.get_tensor(scores_idx)   # Expected shape [1, N]
            num_det = interpreter.get_tensor(num_idx)     # Expected shape [1]

            # Squeeze batch dimension
            boxes = np.squeeze(boxes, axis=0)
            classes = np.squeeze(classes, axis=0)
            scores = np.squeeze(scores, axis=0)
            # num_det might be float; cast to int
            try:
                num = int(np.squeeze(num_det).astype(np.int32))
            except Exception:
                num = boxes.shape[0]

            # 4.2 Interpret Results: process detections
            detections = []
            # Iterate only over detected count (num)
            for i in range(min(num, boxes.shape[0], classes.shape[0], scores.shape[0])):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                # TFLite SSD boxes format: [ymin, xmin, ymax, xmax], normalized [0,1]
                y_min, x_min, y_max, x_max = boxes[i].tolist()
                # Scale to pixel coordinates
                xmin = int(x_min * width)
                ymin = int(y_min * height)
                xmax = int(x_max * width)
                ymax = int(y_max * height)
                # 4.3 Post-processing: clip to valid ranges
                xmin, ymin, xmax, ymax = clip_box(xmin, ymin, xmax, ymax, width, height)

                cls_id = int(classes[i])  # Usually 0-based for COCO SSD
                label = labels[cls_id] if (labels and 0 <= cls_id < len(labels)) else f"class_{cls_id}"
                detections.append({
                    'class_id': cls_id,
                    'label': label,
                    'score': score,
                    'box': [xmin, ymin, xmax, ymax],
                })

            # Prepare per-class detections for pseudo-GT and TP/FP records
            dets_by_class = {}
            for det in detections:
                cid = det['class_id']
                dets_by_class.setdefault(cid, []).append((det['score'], det['box']))

            # Update running mAP statistics (self-consistency estimate)
            for cid, det_list in dets_by_class.items():
                records, num_clusters = per_class_pseudo_gt_and_tp_flags(det_list, iou_thresh=0.5)
                # Append records
                if cid not in per_class_records:
                    per_class_records[cid] = []
                per_class_records[cid].extend(records)
                # Update total pseudo ground truths
                per_class_total_gt[cid] = per_class_total_gt.get(cid, 0) + num_clusters

            running_map = compute_running_map(per_class_records, per_class_total_gt)

            # 4.4 Handle Output: draw boxes and labels; overlay running mAP, write to file
            for det in detections:
                xmin, ymin, xmax, ymax = det['box']
                label = det['label']
                score = det['score']
                # Draw rectangle
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Label text
                text = f"{label}: {score:.2f}"
                # Background for text for readability
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (xmin, max(0, ymin - th - baseline)), (xmin + tw + 2, ymin), (0, 255, 0), -1)
                cv2.putText(frame, text, (xmin + 1, max(0, ymin - baseline - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Overlay running mAP and FPS/inference time
            avg_inf_ms = float(np.mean(inference_times)) if inference_times else 0.0
            map_text = f"mAP(est): {running_map*100:.2f}%   Inference: {avg_inf_ms:.1f} ms"
            cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 250), 2, cv2.LINE_AA)

            writer.write(frame)

        # Phase 4.5 Loop end; done after video ends

    finally:
        # Phase 5: Cleanup
        cap.release()
        writer.release()

    # Print summary
    final_map = compute_running_map(per_class_records, per_class_total_gt)
    print("Processing complete.")
    print(f"Frames processed: {frame_count}")
    if inference_times:
        print(f"Average inference time: {np.mean(inference_times):.2f} ms")
    print(f"Final mAP (self-consistency estimate): {final_map*100:.2f}%")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()