#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target device: Raspberry Pi 4B

This script performs object detection on a single input video using a TFLite SSD model,
draws bounding boxes and labels for detected objects, and writes the annotated video to
the specified output path. It also computes and overlays a running "mAP" proxy (confidence-based)
over the processed frames, given the lack of ground-truth annotations.

Phases implemented per Programming Guideline:
- Phase 1: Setup (imports, paths, labels, interpreter, model details)
- Phase 2: Input Acquisition & Preprocessing Loop (video file reading and preprocessing)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling Loop (detection parsing, thresholding, clipping,
           drawing, running mAP-proxy computation, and video writing)
- Phase 5: Cleanup
"""

import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# ==========================
# Phase 1: Setup
# ==========================

# 1.2 Paths/Parameters (from provided configuration)
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = float('0.5')  # Confidence threshold, converted from string parameter

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1.3 Load Labels (if provided and relevant)
def load_labels(path):
    labels = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    return labels

labels = load_labels(label_path)

# Helper: Convert class id to label string
def get_label_name(class_id):
    if 0 <= class_id < len(labels):
        return labels[class_id]
    else:
        return f'class_{class_id}'

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input shape and dtype
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # Expected [1, H, W, 3]
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Parse expected input size
if len(input_shape) == 4:
    _, in_h, in_w, in_c = input_shape
else:
    raise RuntimeError(f'Unexpected input tensor shape: {input_shape}')

# ==========================
# Phase 2: Input Acquisition & Preprocessing Loop
# (Read a single video file from the given input_path)
# ==========================

# Open input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise FileNotFoundError(f'Unable to open video file: {input_path}')

# Read first frame to get properties
ret, frame = cap.read()
if not ret or frame is None:
    cap.release()
    raise RuntimeError('Failed to read the first frame from the input video.')

frame_h, frame_w = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0  # Fallback if FPS is not available

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f'Unable to open VideoWriter for: {output_path}')

# mAP proxy tracking: collect scores per class across frames
# We use a confidence-based proxy due to lack of ground-truth annotations.
class_scores = {}  # dict: class_id -> list of detection scores

# Utility: color generation per class id
def class_id_to_color(cid):
    # Simple deterministic color from class id
    np.random.seed(cid + 12345)
    color = np.random.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

# Helper: Compute running mAP proxy (mean of per-class mean confidences)
def compute_map_proxy(scores_dict):
    per_class_aps = []
    for cid, scores in scores_dict.items():
        if len(scores) > 0:
            per_class_aps.append(float(np.mean(scores)))
    if len(per_class_aps) == 0:
        return 0.0
    return float(np.mean(per_class_aps))

# Frame processing loop
frame_index = 0

while True:
    # 2.2 Preprocess Data for current frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h))
    input_data = np.expand_dims(resized, axis=0)

    # 2.3 Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = input_data.astype(input_dtype)

    # ==========================
    # Phase 3: Inference
    # ==========================
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # ==========================
    # Phase 4: Output Interpretation & Handling Loop
    # ==========================

    # 4.1 Get Output Tensors
    raw_outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    # 4.2 Interpret Results (identify boxes, classes, scores, num_detections)
    boxes = None
    classes = None
    scores = None
    num_detections = None

    # Robust detection of outputs by shapes/ranges
    for arr in raw_outputs:
        arr_np = np.squeeze(arr)
        arr_shape = arr.shape
        if len(arr_shape) == 3 and arr_shape[-1] == 4:
            # Expect (1, N, 4) for boxes
            boxes = arr[0]
        elif arr_np.size == 1:
            # num_detections
            num_detections = int(round(float(arr_np)))
        elif len(arr_shape) == 2 and arr_shape[0] == 1:
            # Could be scores or classes
            # Scores are usually in [0,1]; classes are ints (as float) > 1.0
            max_val = float(np.max(arr))
            min_val = float(np.min(arr))
            if max_val <= 1.0 and min_val >= 0.0 and scores is None:
                scores = arr[0]
            elif classes is None:
                classes = arr[0]

    # Fallbacks if any are missing
    if boxes is None:
        # Try to find any array with last dim 4
        for arr in raw_outputs:
            if len(arr.shape) >= 2 and arr.shape[-1] == 4:
                boxes = arr[0]
                break
    if classes is None:
        # Identify non-scores array if possible
        for arr in raw_outputs:
            if len(arr.shape) == 2 and arr.shape[0] == 1:
                max_val = float(np.max(arr))
                min_val = float(np.min(arr))
                if not (max_val <= 1.0 and min_val >= 0.0):
                    classes = arr[0]
                    break
    if scores is None:
        for arr in raw_outputs:
            if len(arr.shape) == 2 and arr.shape[0] == 1:
                max_val = float(np.max(arr))
                min_val = float(np.min(arr))
                if max_val <= 1.0 and min_val >= 0.0:
                    scores = arr[0]
                    break
    if boxes is None or classes is None or scores is None:
        # Cannot interpret outputs; write the original frame and continue
        cv2.putText(frame, 'Detection outputs not found', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        # Overlay mAP proxy so far
        map_proxy = compute_map_proxy(class_scores)
        cv2.putText(frame, f'mAP: {map_proxy * 100:.2f}%', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
        writer.write(frame)
    else:
        # Ensure 1D arrays and consistent length
        if len(boxes.shape) == 3:
            boxes = boxes[0]
        if len(classes.shape) == 2:
            classes = classes[0]
        if len(scores.shape) == 2:
            scores = scores[0]

        if num_detections is None:
            num_detections = min(len(scores), len(classes), len(boxes))
        else:
            num_detections = min(num_detections, len(scores), len(classes), len(boxes))

        # 4.3 Post-processing: thresholding, coordinate scaling, and clipping
        for i in range(num_detections):
            score = float(scores[i])
            if score < CONF_THRESHOLD:
                continue

            cid = int(classes[i])  # class id may come as float; cast to int
            y_min, x_min, y_max, x_max = boxes[i]  # normalized coordinates

            # Clip normalized coordinates to [0,1]
            y_min = max(0.0, min(1.0, float(y_min)))
            x_min = max(0.0, min(1.0, float(x_min)))
            y_max = max(0.0, min(1.0, float(y_max)))
            x_max = max(0.0, min(1.0, float(x_max)))

            # Scale to image coordinates
            left = int(round(x_min * frame_w))
            top = int(round(y_min * frame_h))
            right = int(round(x_max * frame_w))
            bottom = int(round(y_max * frame_h))

            # Ensure proper ordering
            left, right = min(left, right), max(left, right)
            top, bottom = min(top, bottom), max(top, bottom)

            # Clip to frame bounds
            left = max(0, min(frame_w - 1, left))
            right = max(0, min(frame_w - 1, right))
            top = max(0, min(frame_h - 1, top))
            bottom = max(0, min(frame_h - 1, bottom))

            # Skip invalid/empty boxes
            if right <= left or bottom <= top:
                continue

            # Update running per-class scores for mAP proxy
            if cid not in class_scores:
                class_scores[cid] = []
            class_scores[cid].append(score)

            # Draw rectangle and label text
            color = class_id_to_color(cid)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label_name = get_label_name(cid)
            caption = f'{label_name}: {score:.2f}'
            # Draw label background for readability
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (left, max(0, top - th - baseline - 4)),
                          (left + tw + 4, top), color, thickness=-1)
            cv2.putText(frame, caption, (left + 2, top - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Compute and overlay running mAP proxy
        map_proxy = compute_map_proxy(class_scores)
        cv2.putText(frame, f'mAP: {map_proxy * 100:.2f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)

        # 4.4 Handle Output: write annotated frame to output video
        writer.write(frame)

    # 4.5 Loop continuation: read next frame or break
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame_index += 1

# ==========================
# Phase 5: Cleanup
# ==========================
cap.release()
writer.release()

# Print final mAP proxy summary
final_map_proxy = compute_map_proxy(class_scores)
print(f'Processing completed. Estimated mAP (confidence-based proxy): {final_map_proxy * 100:.2f}%')
print(f'Annotated video written to: {output_path}')