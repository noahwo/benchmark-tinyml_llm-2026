#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script loads a TFLite SSD MobileNet model and performs object detection on a single input
video file. It writes an output video with bounding boxes and labels overlaid. It also computes
and overlays a proxy mAP value (mean confidence of kept detections) since ground-truth annotations
are not provided for true mAP computation.

Phases implemented per Programming Guideline:
- Phase 1: Setup (imports, paths, load labels, load interpreter, get model details)
- Phase 2: Input Acquisition & Preprocessing Loop (read video file, preprocess frames)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling Loop
  - 4.2: Interpret Results (map classes to labels, structure detections)
  - 4.3: Post-processing (thresholding, coordinate scaling, clipping)
- Phase 5: Cleanup (release resources)
"""

import os
import time
import numpy as np
import cv2

# -----------------------------
# Phase 1: Setup
# -----------------------------
# 1.1 Imports: Interpreter per instructions
from ai_edge_litert.interpreter import Interpreter  # NOTE: as required by the guideline

# 1.2 Paths/Parameters
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD_STR  = 0.5
CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_STR)

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1.3 Load Labels (if provided and relevant)
def load_labels(label_path):
    labels = []
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
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

# Extract input properties
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

if len(input_shape) != 4 or input_shape[-1] != 3:
    raise ValueError(f"Unexpected model input shape: {input_shape}. Expected [1, H, W, 3].")

in_h, in_w = int(input_shape[1]), int(input_shape[2])

# -----------------------------
# Utility functions
# -----------------------------
def preprocess_frame_bgr_to_input(frame_bgr, target_w, target_h, dtype, floating):
    """
    Convert BGR frame to model input tensor:
    - Resize to (target_w, target_h)
    - Convert BGR->RGB
    - Add batch dimension
    - Cast to dtype
    - Normalize if floating model: (x - 127.5) / 127.5
    """
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)
    if floating:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # For quantized models, keep as uint8
        input_data = input_data.astype(dtype, copy=False)
    return input_data

def extract_detection_outputs(interpreter, output_details):
    """
    Extract TFLite SSD detection outputs robustly, handling variations.
    Expected tensors:
      - boxes: [1, num, 4] or [num, 4]
      - classes: [1, num] or [num]
      - scores: [1, num] or [num]
      - num_detections: [1] or scalar
    Returns:
      boxes (np.ndarray [N, 4], normalized [ymin, xmin, ymax, xmax]),
      classes (np.ndarray [N], int),
      scores (np.ndarray [N], float),
      num (int)
    """
    boxes = None
    classes = None
    scores = None
    num = None

    # Collect raw outputs
    raw_outs = []
    for od in output_details:
        arr = interpreter.get_tensor(od['index'])
        # Remove batch dimension if present
        squeezed = np.squeeze(arr)
        raw_outs.append(squeezed)

    # Heuristic assignment
    for arr in raw_outs:
        if arr.ndim == 2 and arr.shape[1] == 4:
            boxes = arr.astype(np.float32, copy=False)
        elif arr.ndim == 1 and arr.size > 1:
            # Distinguish scores vs classes
            # Scores are floats in [0, 1]; classes are typically floats of integer ids
            arr_f32 = arr.astype(np.float32, copy=False)
            if np.all((arr_f32 >= 0.0) & (arr_f32 <= 1.0)):
                scores = arr_f32
            else:
                classes = arr.astype(np.int32, copy=False)
        elif arr.ndim == 0 or arr.size == 1:
            # num_detections may be scalar
            try:
                num = int(float(arr.reshape(-1)[0]))
            except Exception:
                pass

    # If classes still float (e.g., floats close to ints), cast
    if classes is None:
        # Try to find remaining 1D that wasn't assigned as scores
        for arr in raw_outs:
            if arr.ndim == 1 and arr.size > 1:
                arr_f32 = arr.astype(np.float32, copy=False)
                if not np.all((arr_f32 >= 0.0) & (arr_f32 <= 1.0)):
                    classes = arr.astype(np.int32, copy=False)

    # If num is not provided, infer from boxes or scores
    candidate_ns = []
    if boxes is not None:
        candidate_ns.append(boxes.shape[0])
    if scores is not None:
        candidate_ns.append(scores.shape[0])
    if classes is not None:
        candidate_ns.append(classes.shape[0])
    if num is None and candidate_ns:
        num = int(min(candidate_ns))

    # Sanity checks and trimming to num
    if num is None:
        num = 0
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.int32)

    n = min(num, boxes.shape[0], scores.shape[0], classes.shape[0])
    boxes = boxes[:n]
    scores = scores[:n]
    classes = classes[:n]

    return boxes, classes, scores, n

def clip_box(xmin, ymin, xmax, ymax, img_w, img_h):
    xmin = max(0, min(img_w - 1, xmin))
    ymin = max(0, min(img_h - 1, ymin))
    xmax = max(0, min(img_w - 1, xmax))
    ymax = max(0, min(img_h - 1, ymax))
    return int(xmin), int(ymin), int(xmax), int(ymax)

def class_id_to_name(cid, labels_list):
    """
    SSD MobileNet v1 COCO models typically output class IDs starting at 1.
    Map safely to label name if available.
    """
    if labels_list:
        # Try 1-based mapping first
        if 1 <= cid <= len(labels_list):
            return labels_list[cid - 1]
        # Fallback to 0-based if above fails
        if 0 <= cid < len(labels_list):
            return labels_list[cid]
    return f"id:{cid}"

# Proxy mAP: mean of confidences above threshold across processed frames
class ProxymAP:
    def __init__(self):
        self.confidences = []

    def update(self, confs):
        # confs: iterable of confidence scores for kept detections in a frame
        for c in confs:
            if np.isfinite(c):
                self.confidences.append(float(c))

    def value(self):
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))

# -----------------------------
# Phase 2: Input Acquisition & Preprocessing Loop
# -----------------------------
# 2.1 Acquire Input Data: read a single video file from input_path
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open input video: {INPUT_PATH}")

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or np.isnan(fps):
    fps = 30.0  # fallback
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
if not out_writer.isOpened():
    raise RuntimeError(f"Cannot open output video for writing: {OUTPUT_PATH}")

# Initialize stats
proxy_map = ProxymAP()
total_frames = 0
total_kept_dets = 0
t0 = time.time()

# Processing loop for single video file
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break  # end of video

    total_frames += 1

    # 2.2 Preprocess Data
    input_data = preprocess_frame_bgr_to_input(
        frame_bgr, target_w=in_w, target_h=in_h, dtype=input_dtype, floating=floating_model
    )

    # 2.3 Quantization Handling done inside preprocess (normalization for floating model)

    # -----------------------------
    # Phase 3: Inference
    # -----------------------------
    # 3.1 Set Input Tensor
    interpreter.set_tensor(input_index, input_data)

    # 3.2 Run Inference
    interpreter.invoke()

    # -----------------------------
    # Phase 4: Output Interpretation & Handling Loop
    # -----------------------------
    # 4.1 Get Output Tensors
    boxes_norm, classes_raw, scores_raw, det_count = extract_detection_outputs(interpreter, output_details)

    # 4.2 Interpret Results
    # Structure detections, map class IDs to names
    kept_confidences = []
    for i in range(det_count):
        score = float(scores_raw[i]) if i < len(scores_raw) else 0.0
        if score < CONFIDENCE_THRESHOLD or not np.isfinite(score):
            continue  # filter low confidence

        cid = int(classes_raw[i]) if i < len(classes_raw) else -1
        label_name = class_id_to_name(cid, labels)

        # 4.3 Post-processing: scale and clip boxes
        # boxes_norm are [ymin, xmin, ymax, xmax] normalized [0, 1]
        ymin, xmin, ymax, xmax = boxes_norm[i] if i < len(boxes_norm) else (0, 0, 0, 0)
        xmin_abs = xmin * frame_width
        xmax_abs = xmax * frame_width
        ymin_abs = ymin * frame_height
        ymax_abs = ymax * frame_height

        x1, y1, x2, y2 = clip_box(xmin_abs, ymin_abs, xmax_abs, ymax_abs, frame_width, frame_height)

        # Draw bounding box and label on frame
        color = (0, 255, 0)  # green box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness=2)
        label_text = f"{label_name} {score:.2f}"
        # Background rectangle for text
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame_bgr, label_text, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        kept_confidences.append(score)
        total_kept_dets += 1

    # Update proxy mAP (mean confidence of kept detections)
    proxy_map.update(kept_confidences)

    # 4.4 Handle Output: overlay stats and write to output video
    # Compute running FPS
    elapsed = time.time() - t0
    running_fps = total_frames / elapsed if elapsed > 0 else 0.0
    # Overlay information
    overlay_lines = [
        f"Frame: {total_frames}",
        f"Detections(>= {CONFIDENCE_THRESHOLD:.2f}): {len(kept_confidences)}",
        f"mAP (proxy mean conf): {proxy_map.value():.3f}",
        f"FPS: {running_fps:.2f}"
    ]
    y_base = 20
    for li, text in enumerate(overlay_lines):
        cv2.putText(frame_bgr, text, (10, y_base + li * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    out_writer.write(frame_bgr)

# 4.5 Loop ended

# -----------------------------
# Phase 5: Cleanup
# -----------------------------
cap.release()
out_writer.release()

# Final summary print
print("Processing complete.")
print(f"Input video: {INPUT_PATH}")
print(f"Output video: {OUTPUT_PATH}")
print(f"Total frames processed: {total_frames}")
print(f"Total detections kept (score >= {CONFIDENCE_THRESHOLD:.2f}): {total_kept_dets}")
print(f"Proxy mAP (mean confidence of kept detections): {proxy_map.value():.4f}")