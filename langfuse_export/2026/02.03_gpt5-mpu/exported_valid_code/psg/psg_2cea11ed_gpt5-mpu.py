#!/usr/bin/env python3
"""
Object Detection via a video file on Raspberry Pi 4B
- Loads a TFLite SSD MobileNet v1 model using ai_edge_litert Interpreter
- Reads a single video file from input_path
- Performs inference per frame, draws bounding boxes and labels
- Applies post-processing: confidence thresholding, coordinate scaling, clipping
- Writes an output video with detections drawn
- Calculates and overlays a proxy mAP (mean Average Precision) metric over time
  (Proxy mAP is computed without ground truth by using NMS-suppressed predictions
   as pseudo ground-truth per frame; this measures duplicate suppression quality)
Requirements:
- Only standard libraries (os, time, numpy) and cv2 for video processing are used.
- Interpreter import is: from ai_edge_litert.interpreter import Interpreter
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: Interpreter per guideline
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (from configuration)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = float('0.5')  # Confidence Threshold

# Utility: Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def load_labels(label_path):
    """
    Load labels from a given label map file.
    Returns:
        labels (list[str]): list of label names indexed by class id.
    """
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name != '':
                labels.append(name)
    return labels


def letterbox_resize(img, target_size):
    """
    Resize image to target_size without preserving aspect ratio (direct resize).
    For SSD models, direct resize is acceptable.
    Args:
        img (np.ndarray): BGR frame
        target_size (tuple): (width, height)
    Returns:
        resized (np.ndarray): resized image
    """
    target_w, target_h = target_size
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized


def preprocess_frame(frame_bgr, input_shape, floating_model):
    """
    Preprocess BGR frame to match model input tensor requirements.
    - Resize to (input_w,input_h)
    - Convert BGR to RGB
    - Add batch dimension
    - Normalize if floating model: (x - 127.5) / 127.5
    Args:
        frame_bgr (np.ndarray): input frame in BGR
        input_shape (tuple): model input tensor shape (1, H, W, C)
        floating_model (bool): if True, normalize to [-1,1]
    Returns:
        input_data (np.ndarray): preprocessed tensor ready for set_tensor
        resized_rgb (np.ndarray): resized RGB image (H,W,3) for potential debugging
    """
    _, in_h, in_w, in_c = input_shape
    if in_c != 3:
        raise ValueError("Model input channel count is not 3; unsupported input format for this script.")

    resized_bgr = letterbox_resize(frame_bgr, (in_w, in_h))
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    input_data = resized_rgb

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)

    input_data = np.expand_dims(input_data, axis=0)  # add batch dimension
    return input_data, resized_rgb


def extract_detections_from_outputs(outputs):
    """
    Robustly extract boxes, classes, scores, and num_detections from TFLite outputs.
    The function attempts to infer the correct arrays based on shapes/dtypes.
    Args:
        outputs (list[np.ndarray]): raw output tensors as returned by interpreter.get_tensor()
    Returns:
        boxes (np.ndarray): [N,4] float (ymin,xmin,ymax,xmax) normalized 0..1
        classes (np.ndarray): [N] int class ids
        scores (np.ndarray): [N] float confidences 0..1
        num (int): number of detections
    """
    boxes = None
    classes = None
    scores = None
    num = None

    # Collect candidates
    boxes_cand = []
    classes_cand = []
    scores_cand = []
    num_cand = []

    for arr in outputs:
        arr_np = np.array(arr)
        if arr_np.ndim == 3 and arr_np.shape[0] == 1 and arr_np.shape[2] == 4:
            boxes_cand.append(arr_np[0])
        elif arr_np.ndim == 2 and arr_np.shape[0] == 1:
            # Could be classes or scores
            a0 = arr_np[0]
            if a0.dtype.kind == 'f':
                # Likely scores if values are within [0,1]
                maxv = np.max(a0) if a0.size > 0 else 0.0
                minv = np.min(a0) if a0.size > 0 else 0.0
                if (minv >= -1e-3) and (maxv <= 1.0 + 1e-3):
                    scores_cand.append(a0.astype(np.float32))
                else:
                    classes_cand.append(a0.astype(np.float32))
            else:
                classes_cand.append(a0.astype(np.int32))
        elif arr_np.size == 1:
            num_cand.append(int(np.squeeze(arr_np)))
        else:
            # Ignore unknown shapes
            pass

    boxes = boxes_cand[0] if boxes_cand else None
    scores = scores_cand[0] if scores_cand else None

    # Classes: prefer int dtype; if float, cast to int
    if classes_cand:
        if np.issubdtype(classes_cand[0].dtype, np.integer):
            classes = classes_cand[0].astype(np.int32)
        else:
            classes = np.round(classes_cand[0]).astype(np.int32)

    if num_cand:
        num = num_cand[0]

    # Fallback to typical SSD order if inference failed to map
    if boxes is None or scores is None or classes is None or num is None:
        # Attempt to interpret typical order: [boxes, classes, scores, num_detections]
        try:
            boxes = outputs[0][0]
            classes = np.round(outputs[1][0]).astype(np.int32)
            scores = outputs[2][0].astype(np.float32)
            num = int(np.squeeze(outputs[3]))
        except Exception:
            raise RuntimeError("Unable to parse TFLite detection outputs. Check model outputs.")

    n = min(num, boxes.shape[0], classes.shape[0], scores.shape[0])
    return boxes[:n], classes[:n], scores[:n], n


def clip_box_xyxy(box, W, H):
    """
    Clip box to image boundaries.
    Args:
        box (tuple/list/np.ndarray): [x1,y1,x2,y2]
        W (int): image width
        H (int): image height
    Returns:
        (np.ndarray): clipped [x1,y1,x2,y2]
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def class_color(cid):
    """
    Deterministic pseudo-random color for a class id.
    Returns BGR tuple.
    """
    # Use simple hashing with prime multipliers
    r = int((37 * (cid + 1)) % 255)
    g = int((17 * (cid + 1)) % 255)
    b = int((29 * (cid + 1)) % 255)
    # Avoid very dark colors
    r = r + 60 if r < 60 else r
    g = g + 60 if g < 60 else g
    b = b + 60 if b < 60 else b
    return (b, g, r)


def iou_xyxy(box, boxes):
    """
    Compute IoU between a single box and an array of boxes.
    Args:
        box (np.ndarray shape [4]): [x1,y1,x2,y2]
        boxes (np.ndarray shape [N,4]): [x1,y1,x2,y2]
    Returns:
        ious (np.ndarray shape [N]): IoU values
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1 + 1)
    inter_h = np.maximum(0, y2 - y1 + 1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    union = area_box + area_boxes - inter + 1e-6
    ious = inter / union
    return ious


def nms_xyxy(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression for xyxy boxes.
    Args:
        boxes (np.ndarray [N,4])
        scores (np.ndarray [N])
        iou_threshold (float)
    Returns:
        keep_indices (list[int])
    """
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def compute_proxy_map(preds_by_class, gts_by_class, iou_threshold=0.5):
    """
    Compute a proxy mAP using predictions as both the input detections and pseudo ground-truth.
    Pseudo ground-truth per frame/class is produced via NMS to remove duplicates.
    Standard AP computation is then applied across all frames for each class, and averaged.
    Args:
        preds_by_class (dict[int, list[dict]]): class_id -> list of {frame_id:int, box:np.ndarray[4], score:float}
        gts_by_class (dict[int, list[dict]]): class_id -> list of {frame_id:int, box:np.ndarray[4]}
        iou_threshold (float): IoU threshold for matching
    Returns:
        mAP (float): mean Average Precision across all classes that have at least one pseudo GT
    """
    aps = []

    for cid in sorted(set(list(preds_by_class.keys()) + list(gts_by_class.keys()))):
        preds = preds_by_class.get(cid, [])
        gts = gts_by_class.get(cid, [])

        if len(gts) == 0:
            # No pseudo ground-truth for this class; skip from mAP
            continue

        # Prepare GT bookkeeping per frame
        gt_by_frame = {}
        for idx, gt in enumerate(gts):
            fid = gt['frame_id']
            if fid not in gt_by_frame:
                gt_by_frame[fid] = []
            gt_by_frame[fid].append({'box': gt['box'], 'matched': False})

        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        for i, p in enumerate(preds_sorted):
            fid = p['frame_id']
            p_box = p['box']
            matched = False

            if fid in gt_by_frame and len(gt_by_frame[fid]) > 0:
                # Compute IoU with all unmatched GT in the same frame
                gt_boxes = np.array([g['box'] for g in gt_by_frame[fid]])
                matched_flags = np.array([g['matched'] for g in gt_by_frame[fid]], dtype=bool)

                if gt_boxes.size > 0:
                    ious = iou_xyxy(p_box, gt_boxes)
                    # Consider only unmatched
                    ious[matched_flags] = -1.0
                    j = int(np.argmax(ious))
                    max_iou = ious[j]
                    if max_iou >= iou_threshold and not gt_by_frame[fid][j]['matched']:
                        matched = True
                        gt_by_frame[fid][j]['matched'] = True

            if matched:
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        # Cumulative sums
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        rec = tp_cum / (len(gts) + 1e-12)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        # Compute AP using precision envelope
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Integration points where recall changes
        idxs = np.where(mrec[1:] != mrec[:-1])[0]
        ap = 0.0
        for i in idxs:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]

        aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return float(np.mean(aps))


def main():
    # 1.3 Load Labels
    labels = load_labels(LABEL_PATH)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: open the given video file
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_PATH}")

    # Prepare output video writer with same size as input frames
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 25.0  # fallback FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {OUTPUT_PATH}")

    # Accumulators for proxy mAP
    preds_by_class = {}  # cid -> list of {frame_id, box, score}
    gts_by_class = {}    # cid -> list of {frame_id, box} (from NMS of predictions)
    frame_index = 0

    start_time = time.time()
    last_time = start_time

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video

        H, W = frame_bgr.shape[:2]

        # 2.2 Preprocess Data
        input_data, _ = preprocess_frame(frame_bgr, input_shape, floating_model)

        # 2.3 Quantization Handling already done in preprocess (normalize for floating)
        # 2.4 Loop Control: implicit via while loop

        # Phase 3: Inference
        # 3.1 Set Input Tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling
        # 4.1 Get Output Tensor(s)
        raw_outputs = [interpreter.get_tensor(od['index']) for od in output_details]

        # 4.2 Interpret Results
        boxes_norm, classes, scores, n = extract_detections_from_outputs(raw_outputs)

        # Convert normalized boxes to image coordinates and collect detections
        detections_for_frame = []  # for drawing (thresholded)
        preds_for_frame_by_class = {}  # for proxy mAP (all predictions without threshold)

        for i in range(n):
            score = float(scores[i])
            if score < 0.0:
                continue  # invalid score

            cls_id = int(classes[i])
            ymin, xmin, ymax, xmax = boxes_norm[i].tolist()

            # 4.3 Post-processing: thresholding, coordinate scaling, clipping
            x1 = int(xmin * W)
            y1 = int(ymin * H)
            x2 = int(xmax * W)
            y2 = int(ymax * H)

            # Ensure proper ordering
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            box_xyxy = clip_box_xyxy([x1, y1, x2, y2], W, H)

            # Store for drawing (apply confidence threshold)
            if score >= CONF_THRESHOLD:
                detections_for_frame.append({
                    'box': box_xyxy,
                    'score': score,
                    'class_id': cls_id
                })

            # Store all predictions (no threshold) for proxy mAP
            if cls_id not in preds_for_frame_by_class:
                preds_for_frame_by_class[cls_id] = []
            preds_for_frame_by_class[cls_id].append({'frame_id': frame_index, 'box': box_xyxy, 'score': score})

        # Proxy mAP: produce pseudo-GT via NMS per class for the current frame
        gts_for_frame_by_class = {}
        for cid, plist in preds_for_frame_by_class.items():
            if len(plist) == 0:
                continue
            boxes = np.array([p['box'] for p in plist], dtype=np.float32)
            scs = np.array([p['score'] for p in plist], dtype=np.float32)
            keep = nms_xyxy(boxes, scs, iou_threshold=0.5)
            if len(keep) == 0:
                continue
            if cid not in gts_for_frame_by_class:
                gts_for_frame_by_class[cid] = []
            for idx in keep:
                gts_for_frame_by_class[cid].append({'frame_id': frame_index, 'box': boxes[idx]})

        # Append to accumulators
        for cid, plist in preds_for_frame_by_class.items():
            if cid not in preds_by_class:
                preds_by_class[cid] = []
            preds_by_class[cid].extend(plist)

        for cid, glist in gts_for_frame_by_class.items():
            if cid not in gts_by_class:
                gts_by_class[cid] = []
            gts_by_class[cid].extend(glist)

        # Compute running proxy mAP (can be computationally acceptable since detections per frame are small)
        running_map = compute_proxy_map(preds_by_class, gts_by_class, iou_threshold=0.5)

        # 4.4 Handle Output: draw detections on frame and write to output video
        # Draw bounding boxes and labels
        for det in detections_for_frame:
            x1, y1, x2, y2 = det['box'].astype(int)
            score = det['score']
            cid = det['class_id']
            color = class_color(cid)
            label_name = labels[cid] if 0 <= cid < len(labels) else f"id_{cid}"

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label_name}: {score:.2f}"
            # Put text background for readability
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame_bgr, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Overlay running mAP
        map_text = f"Proxy mAP@0.5: {running_map:.3f}"
        cv2.rectangle(frame_bgr, (5, 5), (5 + 210, 30), (0, 0, 0), -1)
        cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame_bgr)

        frame_index += 1

        # 4.5 Loop Continuation: handled by while, will break at end-of-file

    # Final proxy mAP after full video processed
    final_map = compute_proxy_map(preds_by_class, gts_by_class, iou_threshold=0.5)
    elapsed = time.time() - start_time

    # Print summary
    print("Processing complete.")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_index}")
    print(f"Proxy mAP@0.5 (no GT, NMS-based): {final_map:.4f}")
    print(f"Total time: {elapsed:.2f}s  ({(frame_index / max(elapsed, 1e-6)):.2f} FPS)")

    # Phase 5: Cleanup
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()