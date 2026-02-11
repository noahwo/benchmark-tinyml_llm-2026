#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a given video file using a TFLite SSD model
via ai_edge_litert Interpreter. It draws bounding boxes and labels on detected objects,
writes an annotated output video, and computes a running proxy mAP (mean Average Precision)
over the processed frames. Due to the absence of ground-truth annotations, the mAP computed
here is a self-consistency proxy:
- For each frame and each class, only the highest-confidence detection is treated as a TP.
- Other detections for that class in the same frame are treated as FP.
- AP is calculated per class from the precision-recall curve formed by sorting detections by confidence.
- mAP is the mean of AP over classes that had at least one TP during processing.

Note: This proxy mAP is not a substitute for true mAP computed against ground-truth data.
"""

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports (standard libs only as required)
import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter  # literal import as required

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD  = 0.5
METRIC_MIN_SCORE = 0.01  # minimal score to consider a detection in proxy mAP calculation

# Utility: Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def load_labels(label_path):
    """
    1.3 Load Labels (if provided and relevant)
    Reads the label file into a list, stripping whitespace and empty lines.
    """
    labels = []
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    labels.append(line)
    return labels


def get_label_for_class_id(labels, class_id):
    """
    Map class_id to a human-readable label using the labels list.
    Handles potential off-by-one indexing differences between model outputs and label file.
    """
    ci = int(class_id)
    # Direct mapping
    if 0 <= ci < len(labels):
        return labels[ci]
    # Fallback: if outputs are 1-based but labels are 0-based
    if 0 <= (ci - 1) < len(labels):
        return labels[ci - 1]
    # As a last resort, return generic identifier
    return f'id_{ci}'


def color_for_class_id(class_id):
    """
    Generate a pseudo-random but deterministic color for a given class id.
    """
    ci = int(class_id)
    np.random.seed(ci)
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return color


def compute_ap_from_prec_recall(precisions, recalls):
    """
    Compute Average Precision (AP) using the typical interpolation method:
    - Ensure precision is a non-increasing function by setting p[i] = max(p[i], p[i+1], ...)
    - Sum over recall deltas weighted by precision at those points.
    """
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0
    # Append boundary points
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # Make precision non-increasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Integrate area under curve where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def compute_proxy_map(preds_per_class):
    """
    Compute proxy mAP (no ground truth):
    - For each class, we have a list of detections with fields: frame, score, is_tp (True if it's the
      top detection for that class in that frame, False otherwise).
    - Sort detections by score descending, then compute precision-recall based on TP/FP flags.
    - AP is computed per class; mAP is the mean AP over classes that had at least one TP.
    """
    ap_list = []
    for cls_id, dets in preds_per_class.items():
        if not dets:
            continue
        # Sort detections by score descending
        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        # Determine number of positives (unique frames that contributed a TP)
        npos = len({d['frame'] for d in dets_sorted if d['is_tp']})
        if npos == 0:
            # No positives for this class -> skip
            continue
        # Build TP/FP arrays
        tp = np.array([1 if d['is_tp'] else 0 for d in dets_sorted], dtype=np.float32)
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        recalls = tp_cum / float(npos)
        ap = compute_ap_from_prec_recall(precisions, recalls)
        ap_list.append(ap)
    if len(ap_list) == 0:
        return 0.0, 0  # mAP, number of classes contributing
    mAP = float(np.mean(ap_list))
    return mAP, len(ap_list)


def main():
    # 1.3 Load labels (relevant for visualization)
    labels = load_labels(LABEL_PATH)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Extract input shape and properties
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # e.g., [1, H, W, 3]
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # =========================
    # Phase 2: Input Acquisition & Preprocessing Loop
    # =========================

    # 2.1 Acquire Input Data: Read a single video file from the given input_path
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {INPUT_PATH}")
        raise SystemExit(1)

    # Prepare output video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # default fallback
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        raise SystemExit(1)

    # Aggregators for proxy mAP calculation
    # preds_per_class: dict {class_id: [{'frame': frame_idx, 'score': float, 'is_tp': bool}, ...]}
    preds_per_class = {}

    frame_idx = 0
    t_start = time.time()

    # 2.4 Loop Control: Iterate through all frames in the single input video
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # 2.2 Preprocess Data:
        # - Resize to model input dimensions
        # - Convert BGR (OpenCV) to RGB if needed by model
        # - Expand dims to [1, H, W, 3]
        # - Cast to required dtype and normalize if floating model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        input_data = np.expand_dims(resized, axis=0)

        # Ensure dtype
        if floating_model:
            # 2.3 Quantization Handling: normalize to [-1, 1] for floating models
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # uint8 quantized model expects 0..255
            input_data = np.uint8(input_data)

        # =========================
        # Phase 3: Inference (Run per preprocessed input)
        # =========================

        # 3.1 Set Input Tensor(s)
        interpreter.set_tensor(input_index, input_data)

        # 3.2 Run Inference
        interpreter.invoke()

        # =========================
        # Phase 4: Output Interpretation & Handling Loop
        # =========================

        # 4.1 Get Output Tensor(s)
        # Try to identify boxes, classes, scores, and num detections
        boxes = None
        classes = None
        scores = None
        num = None

        # Retrieve all output tensors
        out_tensors = []
        for od in output_details:
            out_tensors.append(interpreter.get_tensor(od['index']))

        # Identify tensors by shape and value ranges
        for out in out_tensors:
            arr = np.squeeze(out)
            if arr.ndim == 2 and arr.shape[1] == 4:
                # Shape [N, 4] or [1, N, 4]; normalize to [N, 4]
                boxes = arr
            elif arr.ndim == 1 and arr.size > 4:
                # Could be classes or scores; distinguish by value range
                maxv = float(np.max(arr)) if arr.size > 0 else 0.0
                minv = float(np.min(arr)) if arr.size > 0 else 0.0
                if maxv <= 1.0001 and minv >= 0.0:
                    scores = arr
                else:
                    classes = arr
            elif arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
                num = int(np.round(float(arr)))

        # Fallback for common TFLite SSD order if above heuristic fails
        if boxes is None or classes is None or scores is None:
            # Try the conventional ordering: [boxes, classes, scores, num]
            try:
                b = np.squeeze(out_tensors[0])
                c = np.squeeze(out_tensors[1])
                s = np.squeeze(out_tensors[2])
                n = int(np.round(float(np.squeeze(out_tensors[3]))))
                if b.ndim == 2 and b.shape[1] == 4:
                    boxes = b
                if c.ndim == 1:
                    classes = c
                if s.ndim == 1:
                    scores = s
                num = n
            except Exception:
                pass

        # Validate outputs
        if boxes is None or classes is None or scores is None:
            # If outputs are still not recognized, skip frame gracefully
            cv2.putText(frame, "Output parsing error", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(frame)
            frame_idx += 1
            continue

        # Determine number of detections
        if num is None:
            num = min(len(scores), len(classes), len(boxes))
        else:
            num = min(num, len(scores), len(classes), len(boxes))

        # 4.2 Interpret Results:
        # Convert normalized box coordinates to pixel coordinates and map class IDs to labels
        h, w = frame.shape[0], frame.shape[1]
        detections_for_frame = []
        for i in range(num):
            score = float(scores[i])
            cls_id = int(classes[i])

            # Ignore extremely low scores for speed in both drawing and metric
            if score < METRIC_MIN_SCORE:
                continue

            # TFLite SSD boxes are in [ymin, xmin, ymax, xmax], normalized [0,1]
            ymin, xmin, ymax, xmax = boxes[i]
            # 4.3 Post-processing: clip coordinates to valid ranges
            ymin = max(0.0, min(1.0, float(ymin)))
            xmin = max(0.0, min(1.0, float(xmin)))
            ymax = max(0.0, min(1.0, float(ymax)))
            xmax = max(0.0, min(1.0, float(xmax)))

            # Scale to pixel coordinates
            x1 = int(round(xmin * w))
            y1 = int(round(ymin * h))
            x2 = int(round(xmax * w))
            y2 = int(round(ymax * h))

            # Ensure proper ordering and clipping
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue  # discard degenerate boxes

            label_text = get_label_for_class_id(labels, cls_id) if labels else f'class_{cls_id}'
            detections_for_frame.append({
                'class_id': cls_id,
                'score': score,
                'bbox': (x1, y1, x2, y2),
                'label': label_text
            })

        # Build per-class organization for the current frame (for proxy mAP)
        per_class_scores = {}
        for det in detections_for_frame:
            cid = det['class_id']
            per_class_scores.setdefault(cid, [])
            per_class_scores[cid].append(det['score'])

        # For each class in this frame, identify the top score as TP; others as FP
        used_tp_for_frame_class = set()
        for det in detections_for_frame:
            cid = det['class_id']
            score = det['score']
            # Determine top score for this class in this frame
            top_score = max(per_class_scores[cid]) if len(per_class_scores[cid]) > 0 else score
            is_tp = False
            if (frame_idx, cid) not in used_tp_for_frame_class and abs(score - top_score) < 1e-12:
                # First occurrence of top score for this (frame, class) pair -> TP
                is_tp = True
                used_tp_for_frame_class.add((frame_idx, cid))
            preds_per_class.setdefault(cid, [])
            preds_per_class[cid].append({
                'frame': frame_idx,
                'score': score,
                'is_tp': is_tp
            })

        # 4.4 Handle Output: Draw detections and overlay mAP, then write frame to output video
        draw_count = 0
        for det in detections_for_frame:
            if det['score'] < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = det['bbox']
            color = color_for_class_id(det['class_id'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            caption = f"{det['label']}: {det['score']:.2f}"
            # Put text with background for readability
            (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), color, -1)
            cv2.putText(frame, caption, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            draw_count += 1

        # Compute running proxy mAP and overlay
        mAP_value, contributing_classes = compute_proxy_map(preds_per_class)
        info_lines = [
            f"Detections (>= {CONFIDENCE_THRESHOLD:.2f}): {draw_count}",
            f"Proxy mAP (no GT): {mAP_value:.3f} over {contributing_classes} classes",
            f"Model Input: {in_w}x{in_h}, dtype: {'float32' if floating_model else 'uint8'}"
        ]
        # Draw info box
        padding = 5
        line_h = 18
        box_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in info_lines) + 2 * padding
        box_h = line_h * len(info_lines) + 2 * padding
        cv2.rectangle(frame, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
        for idx, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 10 + (idx + 1) * line_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Write annotated frame to output
        writer.write(frame)
        frame_idx += 1

    # =========================
    # Phase 5: Cleanup
    # =========================
    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    final_mAP, final_classes = compute_proxy_map(preds_per_class)
    print("Processing complete.")
    print(f"Total frames processed: {frame_idx}")
    print(f"Elapsed time (s): {elapsed:.2f}")
    print(f"Proxy mAP (no ground truth): {final_mAP:.4f} over {final_classes} classes")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()