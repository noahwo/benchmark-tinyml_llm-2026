#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application: Object Detection via a video file
Description:
- Reads a single video file from the given input_path.
- Runs SSD MobileNet v1 TFLite model inference using ai_edge_litert Interpreter.
- Outputs a video file with rectangles drawn on detected objects, labels, and an on-frame
  running mAP (proxy) metric. Also prints a final proxy mAP summary.

Phases implemented per Programming Guidelines:
- Phase 1: Setup (imports, paths, labels, interpreter, I/O details)
- Phase 2: Input Acquisition & Preprocessing Loop (explicitly implemented)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling (explicit 4.2 & 4.3 implemented)
- Phase 5: Cleanup

Note:
- Only standard libraries plus numpy and cv2 are used for video/image processing.
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
    Load labels from a text file into a list, stripping whitespace and skipping empty lines.
    """
    labels = []
    if os.path.isfile(label_file_path):
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    labels.append(line)
    return labels

def create_video_writer(out_path, fps, frame_w, frame_h):
    """
    Create a cv2.VideoWriter ensuring parent directories exist.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(frame_w), int(frame_h)))
    return writer

def color_for_class(class_id):
    """
    Deterministic BGR color for a class id.
    """
    rng = np.random.RandomState(int(class_id) * 2654435761 % (2**32))
    color = tuple(int(c) for c in rng.randint(0, 255, size=3))
    return color

def clip_bbox_to_frame(b, width, height):
    """
    Clip bounding box to image boundaries; input b=(ymin,xmin,ymax,xmax) in absolute pixels.
    Returns ints: (ymin,xmin,ymax,xmax).
    """
    ymin, xmin, ymax, xmax = b
    ymin = max(0, min(int(round(ymin)), height - 1))
    xmin = max(0, min(int(round(xmin)), width - 1))
    ymax = max(0, min(int(round(ymax)), height - 1))
    xmax = max(0, min(int(round(xmax)), width - 1))
    # Ensure proper ordering after clipping
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    return ymin, xmin, ymax, xmax

def squeeze_out(arr):
    """
    Remove leading batch dims of size 1 and trailing singleton dims where appropriate.
    Keeps 2D for boxes and 1D for classes/scores.
    """
    a = arr
    # Remove leading batch dimension if 1
    while a.ndim > 0 and a.shape[0] == 1 and a.ndim > 1:
        a = a[0]
    # If still has shape like [1, N], squeeze to [N]
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    return a

def parse_detection_outputs(output_details, output_arrays):
    """
    Parse outputs from SSD-style TFLite detection model.

    Returns:
      boxes: np.ndarray of shape [N,4] (ymin, xmin, ymax, xmax)
      classes: np.ndarray of shape [N] (float/int class ids)
      scores: np.ndarray of shape [N] (float confidences in [0,1])
      num: int, number of valid detections suggested by the model
    """
    # Squeeze and collect
    arrays = [squeeze_out(a) for a in output_arrays]

    boxes = None
    classes = None
    scores = None
    num = None

    # First, try to use names if present
    names = []
    try:
        names = [od.get('name', '') for od in output_details]
    except Exception:
        names = [''] * len(output_details)

    # Map by typical names when available
    for a, n in zip(arrays, names):
        n_low = (n or '').lower()
        if 'box' in n_low or 'bbox' in n_low or (a.ndim == 2 and a.shape[-1] == 4):
            if a.ndim == 2 and a.shape[-1] == 4:
                boxes = a
        if ('class' in n_low or 'label' in n_low) and a.ndim == 1:
            classes = a
        if ('score' in n_low or 'confidence' in n_low) and a.ndim == 1:
            scores = a
        if 'num' in n_low and a.size == 1:
            try:
                num = int(round(float(np.squeeze(a))))
            except Exception:
                pass

    # Heuristic fallback if any missing
    if boxes is None:
        for a in arrays:
            if a.ndim == 2 and a.shape[-1] == 4:
                boxes = a
                break

    one_d = [a for a in arrays if a.ndim == 1 and a.size > 0]
    if scores is None:
        # scores expected to be in [0,1]
        cand = None
        best_range_fit = -1
        for a in one_d:
            amin, amax = float(np.min(a)), float(np.max(a))
            # Score arrays should largely be within [0,1]
            comp = -abs(0.0 - amin) - abs(1.0 - max(min(amax, 1.0), 0.0))
            if comp > best_range_fit:
                best_range_fit = comp
                cand = a
        scores = cand

    if classes is None and len(one_d) > 0:
        # Choose the 1-D array different from scores with more integer-like values
        cand_list = [a for a in one_d if a is not scores]
        if len(cand_list) == 1:
            classes = cand_list[0]
        elif len(cand_list) > 1:
            best_int_like = -1.0
            best = None
            for a in cand_list:
                frac = np.mod(a, 1.0)
                int_like = float(np.mean(np.abs(frac) < 1e-3))
                if int_like > best_int_like:
                    best_int_like = int_like
                    best = a
            classes = best

    if num is None:
        # Infer num from the smallest consistent length
        lengths = []
        if boxes is not None:
            lengths.append(int(boxes.shape[0]))
        if classes is not None:
            lengths.append(int(classes.shape[0]))
        if scores is not None:
            lengths.append(int(scores.shape[0]))
        num = int(min(lengths)) if len(lengths) > 0 else 0

    # Ensure proper shapes and consistent lengths
    if boxes is not None and boxes.ndim != 2:
        boxes = np.reshape(boxes, (-1, 4)) if boxes.size % 4 == 0 else None

    # Final safety trims based on computed num
    def safe_slice_1d(a, n):
        if a is None:
            return None
        return a[:min(n, a.shape[0])] if a.ndim == 1 else a

    def safe_slice_2d(a, n):
        if a is None:
            return None
        return a[:min(n, a.shape[0]), :] if a.ndim == 2 else a

    boxes = safe_slice_2d(boxes, num)
    classes = safe_slice_1d(classes, num)
    scores = safe_slice_1d(scores, num)

    # Recompute n as the minimum available length to prevent OOB
    lengths = []
    if boxes is not None and boxes.ndim == 2:
        lengths.append(int(boxes.shape[0]))
    if classes is not None and classes.ndim == 1:
        lengths.append(int(classes.shape[0]))
    if scores is not None and scores.ndim == 1:
        lengths.append(int(scores.shape[0]))
    n_final = int(min(lengths)) if len(lengths) > 0 else 0

    return boxes, classes, scores, n_final

def main():
    # 1.2 Paths/Parameters (provided)
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    CONF_THRESHOLD = float('0.5')

    # 1.3 Load Labels
    labels = load_labels(label_path)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Single input expected: [1, H, W, C]
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    in_h, in_w, in_c = int(input_shape[1]), int(input_shape[2]), int(input_shape[3])

    # Quantization params if present
    input_scale, input_zero_point = (1.0, 0)
    try:
        q = input_details[0].get('quantization', (1.0, 0))
        if isinstance(q, (tuple, list)) and len(q) == 2:
            input_scale, input_zero_point = q
            input_scale = 1.0 if input_scale is None else input_scale
            input_zero_point = 0 if input_zero_point is None else input_zero_point
    except Exception:
        pass

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: open the specific video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'ERROR: Failed to open input video: {input_path}')
        return

    # Input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and not np.isnan(fps) and fps > 0 else 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0 else -1

    # Create output writer
    writer = create_video_writer(output_path, fps, frame_w, frame_h)

    # Stats for mAP (proxy): per-class list of confidences above threshold
    class_confidences = {}

    frame_idx = 0
    last_print_time = time.time()

    print('Starting inference on video...')
    print(f'Input: {input_path}')
    print(f'Output: {output_path}')
    print(f'Model: {model_path}')
    print(f'Labels: {label_path} (loaded {len(labels)} labels)')
    print(f'Confidence threshold: {CONF_THRESHOLD:.2f}')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            frame_idx += 1

            original_frame = frame  # will annotate directly

            # 2.2 Preprocess Data: resize to model input, convert BGR->RGB
            resized = cv2.resize(original_frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # 2.3 Quantization Handling
            if input_dtype == np.float32:
                # Normalize to [-1, 1] as common for SSD MobileNet
                input_data = (np.float32(rgb) - 127.5) / 127.5
            elif input_dtype == np.uint8:
                input_data = np.uint8(rgb)
            elif input_dtype == np.int8:
                # Quantize from normalized float using scale/zero-point when provided
                float_in = (np.float32(rgb) - 127.5) / 127.5
                scale = input_scale if input_scale not in (None, 0) else 1.0
                zero = input_zero_point if input_zero_point is not None else 0
                q = np.round(float_in / scale + zero)
                q = np.clip(q, -128, 127)
                input_data = q.astype(np.int8)
            else:
                input_data = rgb.astype(input_dtype)

            # Add batch dimension [1, H, W, C]
            input_data = np.expand_dims(input_data, axis=0)

            # Phase 3: Inference
            interpreter.set_tensor(input_index, input_data)
            t0 = time.time()
            interpreter.invoke()
            infer_time_ms = (time.time() - t0) * 1000.0

            # Phase 4: Output Interpretation & Handling
            # 4.1 Get Output Tensor(s)
            output_arrays = []
            for od in output_details:
                out = interpreter.get_tensor(od['index'])
                output_arrays.append(out)

            # 4.2 Interpret Results: parse SSD outputs (boxes/classes/scores/num)
            boxes, classes, scores, num = parse_detection_outputs(output_details, output_arrays)

            # 4.3 Post-processing: thresholding, coordinate scaling, clipping
            kept = []
            if boxes is not None and classes is not None and scores is not None and num > 0:
                # Decide if boxes are normalized or absolute by checking range
                # If typical SSD output: normalized [0,1]. If max coord > 2, treat as absolute.
                max_box_val = float(np.max(np.abs(boxes))) if boxes.size > 0 else 0.0
                boxes_are_normalized = not (max_box_val > 2.0)

                N = int(min(num, boxes.shape[0], classes.shape[0], scores.shape[0]))
                for i in range(N):
                    score = float(scores[i])
                    if score < CONF_THRESHOLD:
                        continue

                    # Class id (may be float from model); round then int
                    cls_id_raw = int(round(float(classes[i])))

                    y_min, x_min, y_max, x_max = boxes[i]
                    if boxes_are_normalized:
                        y_min *= frame_h
                        y_max *= frame_h
                        x_min *= frame_w
                        x_max *= frame_w

                    y_min, x_min, y_max, x_max = clip_bbox_to_frame((y_min, x_min, y_max, x_max), frame_w, frame_h)
                    kept.append((cls_id_raw, score, (x_min, y_min, x_max, y_max)))

                    # Accumulate for proxy mAP
                    class_confidences.setdefault(cls_id_raw, []).append(score)

            # Draw detections with labels
            for (cls_id, score, (x_min, y_min, x_max, y_max)) in kept:
                color = color_for_class(cls_id)
                cv2.rectangle(original_frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Resolve label name
                if 0 <= cls_id < len(labels):
                    label_name = labels[cls_id]
                elif 0 <= (cls_id - 1) < len(labels):
                    # Some SSD label maps are 1-based (background at 0)
                    label_name = labels[cls_id - 1]
                else:
                    label_name = f'id_{cls_id}'

                caption = f'{label_name}: {score:.2f}'
                (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_text_top = max(0, y_min - th - bl - 4)
                cv2.rectangle(original_frame, (x_min, y_text_top), (x_min + tw + 6, y_min), color, thickness=-1)
                cv2.putText(original_frame, caption, (x_min + 3, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Compute running mAP (proxy): mean of per-class mean confidences
            ap_vals = []
            for _, conf_list in class_confidences.items():
                if len(conf_list) > 0:
                    ap_vals.append(float(np.mean(conf_list)))
            mAP_proxy = float(np.mean(ap_vals)) if len(ap_vals) > 0 else 0.0

            # 4.4 Handle Output: overlay info and write to output video
            info1 = f'Frame: {frame_idx}/{total_frames if total_frames > 0 else "?"}  Inference: {infer_time_ms:.1f} ms'
            info2 = f'mAP (proxy): {mAP_proxy:.3f}  Detections: {len(kept)}  Thresh: {CONF_THRESHOLD:.2f}'
            cv2.putText(original_frame, info1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
            cv2.putText(original_frame, info2, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)

            writer.write(original_frame)

            # 4.5 Progress logging (periodic)
            now = time.time()
            if now - last_print_time > 2.0:
                print(f'Processed frame {frame_idx} | {len(kept)} detections | mAP (proxy): {mAP_proxy:.3f}')
                last_print_time = now

    finally:
        # Phase 5: Cleanup
        cap.release()
        writer.release()

    # Final report
    final_ap_vals = []
    for _, conf_list in class_confidences.items():
        if len(conf_list) > 0:
            final_ap_vals.append(float(np.mean(conf_list)))
    final_map_proxy = float(np.mean(final_ap_vals)) if len(final_ap_vals) > 0 else 0.0

    print('Inference complete.')
    print(f'Approximate mAP (proxy, mean per-class confidence over kept detections): {final_map_proxy:.4f}')
    print(f'Annotated video saved to: {output_path}')


if __name__ == '__main__':
    main()

# -------------------------------------------------------------------------
# REPORT OF THE FIX OF THE LAST ERROR:
# The previous script crashed with IndexError when iterating detections:
#   for i in range(num): cls_id_raw = int(round(float(classes[i])))
# because 'num' exceeded the actual length of the 'classes' array.
#
# Fixes implemented:
# 1) Robust output parsing (parse_detection_outputs) that:
#    - Squeezes shapes, identifies boxes/classes/scores using names and heuristics.
#    - Computes 'num' and then re-slices arrays safely.
#    - Returns the final N as the minimum consistent length across outputs.
# 2) In post-processing, iterate using N = min(num, len(boxes), len(classes), len(scores))
#    to prevent out-of-bounds indexing.
# These changes ensure safe indexing and prevent the IndexError.
# -------------------------------------------------------------------------