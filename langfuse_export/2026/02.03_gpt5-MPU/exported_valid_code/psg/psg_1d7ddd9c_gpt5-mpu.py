#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection using a TFLite SSD MobileNet v1 model on a single input video file.
It writes an output video with bounding boxes and labels drawn for detected objects and overlays a
per-frame "mAP (vs previous frame)" value computed by comparing current detections against detections
from the previous frame (as pseudo ground truth). This provides an internal consistency metric when
ground truth annotations are unavailable.

Phases implemented per Programming Guideline:
- Phase 1: Setup (imports, paths, load labels, load interpreter, get model details)
- Phase 2: Input Acquisition & Preprocessing Loop (video reading, resizing, normalization/quantization)
- Phase 3: Inference (set tensors, invoke)
- Phase 4: Output Interpretation & Handling Loop (parse detections, thresholding, scaling, clipping, draw)
- Phase 5: Cleanup (release video resources)

Configuration Parameters:
- model_path: 'models/ssd-mobilenet_v1/detect.tflite'
- label_path: 'models/ssd-mobilenet_v1/labelmap.txt'
- input_path: 'data/object_detection/sheeps.mp4'
- output_path: 'results/object_detection/test_results/sheeps_detections.mp4'
- confidence_threshold: 0.5
"""

import os
import time
import numpy as np
import cv2

# Phase 1.1: Imports (Interpreter import must be literal per guideline)
try:
    from ai_edge_litert.interpreter import Interpreter
except Exception as e:
    print("ERROR: Failed to import Interpreter from ai_edge_litert.interpreter.")
    print("Details:", str(e))
    raise SystemExit(1)


# ----------------------------- Utility Functions -----------------------------

def load_labels(label_file_path):
    """
    Phase 1.3: Load Labels.
    Reads the label file line-by-line and returns a list of label strings.
    Empty lines are ignored.
    """
    labels = []
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name != "":
                    labels.append(name)
    except FileNotFoundError:
        print(f"WARNING: Label file not found at '{label_file_path}'. Proceeding without labels.")
    return labels


def preprocess_frame(frame_bgr, input_shape, input_dtype, input_quant):
    """
    Phase 2.2 and 2.3: Preprocess Data and Quantization Handling.
    - Resize to model input size.
    - Convert BGR to RGB.
    - Expand dims to [1, H, W, 3].
    - If floating model: normalize: (x - 127.5) / 127.5
    - If int8 quantized: apply scale/zero_point.
    - If uint8: pass through.
    """
    _, in_h, in_w, in_c = input_shape  # Typically [1, H, W, 3]
    assert in_c == 3, "Expected 3-channel input."

    # Resize
    resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_NEAREST)
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Expand dims
    input_data = np.expand_dims(rgb, axis=0)

    # Cast/normalize/quantize according to dtype
    floating_model = (input_dtype == np.float32)

    if floating_model:
        # Normalize to [-1, 1]
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Handle quantized types
        if input_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        elif input_dtype == np.int8:
            # Apply quantization parameters if available
            scale, zero_point = input_quant
            if scale is None or scale == 0:
                # Fallback: just cast
                input_data = input_data.astype(np.int8)
            else:
                # Quantize from [0,255] uint8-like to int8 using scale and zp
                input_data = input_data.astype(np.float32)
                input_data = input_data / 255.0  # normalize to [0,1] before quant
                input_data = input_data / scale + zero_point
                input_data = np.round(input_data).astype(np.int8)
        else:
            # Unexpected dtype; do a safe cast
            input_data = input_data.astype(input_dtype)

    return input_data


def dequantize_if_needed(tensor, tensor_detail):
    """
    Dequantize output tensor if it has quantization parameters and is not float32.
    """
    if tensor_detail.get('dtype', None) == np.float32:
        return tensor
    # Try to get quantization parameters
    quant = tensor_detail.get('quantization', (0.0, 0))
    scale, zero_point = (quant if isinstance(quant, (tuple, list)) and len(quant) == 2 else (0.0, 0))
    if scale is not None and scale != 0:
        # Dequantize to float32
        return (tensor.astype(np.float32) - float(zero_point)) * float(scale)
    # If no valid quantization info, cast to float32
    return tensor.astype(np.float32)


def clip_bbox(xmin, ymin, xmax, ymax, width, height):
    """
    Phase 4.3: Bounding box clipping to frame boundaries.
    """
    xmin = max(0, min(int(xmin), width - 1))
    ymin = max(0, min(int(ymin), height - 1))
    xmax = max(0, min(int(xmax), width - 1))
    ymax = max(0, min(int(ymax), height - 1))
    # Ensure proper ordering
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def iou(box_a, box_b):
    """
    Compute IoU between two boxes given as [x1, y1, x2, y2].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_ap_for_class(preds, gts, iou_thresh=0.5):
    """
    Compute Average Precision (AP) for a single class.
    preds: list of tuples (score, bbox) where bbox is [x1, y1, x2, y2]
    gts: list of bboxes [x1, y1, x2, y2]
    Returns AP as float in [0,1].
    """
    num_gts = len(gts)
    if num_gts == 0:
        # Undefined AP if no GT; caller should skip this class from mAP
        return None

    if len(preds) == 0:
        return 0.0

    # Sort predictions by descending score
    preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
    gt_used = np.zeros((num_gts,), dtype=bool)

    tp = np.zeros((len(preds_sorted),), dtype=np.float32)
    fp = np.zeros((len(preds_sorted),), dtype=np.float32)

    # Greedy matching
    for i, (score, pb) in enumerate(preds_sorted):
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gts):
            if gt_used[j]:
                continue
            iou_val = iou(pb, gb)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[i] = 1.0
            gt_used[best_j] = True
        else:
            fp[i] = 1.0

    # Precision-Recall curve
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(num_gts, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # Compute AP as area under precision envelope
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Sum over recall steps where it changes
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)


def compute_map_against_previous(current_dets, prev_dets, iou_thresh=0.5):
    """
    Compute mAP for current frame detections against previous frame detections (pseudo-GT).
    - current_dets: list of dicts {class_id:int, score:float, bbox:[x1,y1,x2,y2]}
    - prev_dets: list of dicts, same structure
    Returns:
    - mAP (float) if computable, else None when no GT exists for any class.
    """
    # Organize detections by class
    preds_by_class = {}
    gts_by_class = {}
    for d in current_dets:
        preds_by_class.setdefault(d['class_id'], []).append((d['score'], d['bbox']))
    for g in prev_dets:
        gts_by_class.setdefault(g['class_id'], []).append(g['bbox'])

    aps = []
    all_classes = set(list(preds_by_class.keys()) + list(gts_by_class.keys()))
    for cid in all_classes:
        preds = preds_by_class.get(cid, [])
        gts = gts_by_class.get(cid, [])
        ap = compute_ap_for_class(preds, gts, iou_thresh=iou_thresh)
        if ap is not None:
            aps.append(ap)

    if len(aps) == 0:
        return None
    return float(np.mean(aps))


def draw_detections(frame, detections, labels):
    """
    Draw bounding boxes and labels onto the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cid = det['class_id']
        score = det['score']
        label_name = None
        if labels and 0 <= cid < len(labels):
            label_name = labels[cid]
        else:
            label_name = f"id:{cid}"

        color = (0, 255, 0)  # Green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_name}: {score:.2f}"
        # Text background
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ----------------------------- Main Application ------------------------------

def main():
    # Phase 1.2: Paths/Parameters from Configuration
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = 0.5

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Phase 1.3: Load Labels
    labels = load_labels(label_path)

    # Phase 1.4: Load Interpreter
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'.")
        raise SystemExit(1)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Phase 1.5: Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        print("WARNING: Model expects more than one input tensor; using the first.")

    in_detail = input_details[0]
    input_index = in_detail['index']
    input_shape = in_detail['shape']
    input_dtype = in_detail['dtype']
    input_quant = in_detail.get('quantization', (None, None))

    # Identify output tensors: boxes, classes, scores, num
    # Phase 4.1 will use these indices to retrieve outputs
    out_indices = {'boxes': None, 'classes': None, 'scores': None, 'num': None}
    for od in output_details:
        od_shape = od['shape']
        if len(od_shape) == 3 and od_shape[-1] == 4:
            out_indices['boxes'] = od['index']
        elif len(od_shape) == 2 and od_shape[-1] >= 1:
            # Could be classes or scores; need dtype heuristic
            if od.get('dtype') in (np.float32, np.float16):
                out_indices['scores'] = od['index']
            else:
                out_indices['classes'] = od['index']
        elif len(od_shape) == 1 and od_shape[0] == 1:
            out_indices['num'] = od['index']

    # Fallback if not identified by heuristic (handle typical SSD order)
    if any(v is None for v in out_indices.values()):
        # Attempt standard order by assuming 4 outputs
        if len(output_details) >= 4:
            out_indices['boxes'] = output_details[0]['index'] if out_indices['boxes'] is None else out_indices['boxes']
            out_indices['classes'] = output_details[1]['index'] if out_indices['classes'] is None else out_indices['classes']
            out_indices['scores'] = output_details[2]['index'] if out_indices['scores'] is None else out_indices['scores']
            out_indices['num'] = output_details[3]['index'] if out_indices['num'] is None else out_indices['num']

    # Phase 2.1: Acquire Input Data (video file)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Unable to open input video: '{input_path}'.")
        raise SystemExit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # reasonable default if unavailable
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: Unable to open output video writer: '{output_path}'.")
        cap.release()
        raise SystemExit(1)

    # Tracking previous frame detections for pseudo-GT mAP computation
    prev_detections = []
    running_map_sum = 0.0
    running_map_count = 0
    frame_count = 0

    # Phase 2.4: Loop Control - process entire video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Phase 2.2 & 2.3: Preprocess
        input_data = preprocess_frame(frame, input_shape, input_dtype, input_quant)

        # Phase 3.1: Set Input Tensor
        interpreter.set_tensor(input_index, input_data)

        # Phase 3.2: Run Inference
        start_infer = time.time()
        interpreter.invoke()
        infer_time_ms = (time.time() - start_infer) * 1000.0

        # Phase 4.1: Get Output Tensors
        # Retrieve and dequantize if necessary
        boxes_raw = interpreter.get_tensor(out_indices['boxes'])
        classes_raw = interpreter.get_tensor(out_indices['classes'])
        scores_raw = interpreter.get_tensor(out_indices['scores'])
        num_raw = interpreter.get_tensor(out_indices['num'])

        # Dequantize outputs if needed
        # Identify output details dict by index for dequantization
        out_detail_by_index = {od['index']: od for od in output_details}
        boxes = dequantize_if_needed(boxes_raw, out_detail_by_index[out_indices['boxes']])
        classes = dequantize_if_needed(classes_raw, out_detail_by_index[out_indices['classes']])
        scores = dequantize_if_needed(scores_raw, out_detail_by_index[out_indices['scores']])
        num_det = int(np.squeeze(num_raw).astype(np.int32))

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores).astype(np.float32)

        # Phase 4.2: Interpret Results - extract, map labels, prepare detections
        detections = []
        for i in range(num_det):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            # SSD boxes are y_min, x_min, y_max, x_max normalized [0,1]
            y_min, x_min, y_max, x_max = boxes[i]
            # Phase 4.3: Post-processing - scale to pixel coords and clip
            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)
            x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, width, height)

            cid = int(classes[i])
            detections.append({
                'class_id': cid,
                'score': score,
                'bbox': [x1, y1, x2, y2]
            })

        # Draw detections (labels if available)
        draw_detections(frame, detections, labels)

        # Compute per-frame mAP against previous frame detections (pseudo-GT)
        frame_map = compute_map_against_previous(detections, prev_detections, iou_thresh=0.5)
        if frame_map is not None:
            running_map_sum += frame_map
            running_map_count += 1

        # Overlay metrics text (mAP and inference time)
        if frame_map is not None:
            map_text = f"mAP (vs prev): {frame_map*100.0:.2f}%"
        else:
            map_text = "mAP (vs prev): N/A"
        inf_text = f"Inference: {infer_time_ms:.1f} ms"

        cv2.putText(frame, map_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, inf_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2, cv2.LINE_AA)

        # Phase 4.4: Handle Output - write frame to output video
        writer.write(frame)

        # Phase 4.5: Loop continuation - update previous detections
        prev_detections = detections

    # Phase 5.1: Cleanup
    cap.release()
    writer.release()

    # Final summary
    if running_map_count > 0:
        overall_map = running_map_sum / running_map_count
        print(f"Processed {frame_count} frames.")
        print(f"Average per-frame mAP (vs previous frame): {overall_map*100.0:.2f}% over {running_map_count} evaluable frames.")
    else:
        print(f"Processed {frame_count} frames.")
        print("Average per-frame mAP could not be computed (no evaluable frames with previous detections).")

    print(f"Output video written to: {output_path}")


if __name__ == "__main__":
    main()