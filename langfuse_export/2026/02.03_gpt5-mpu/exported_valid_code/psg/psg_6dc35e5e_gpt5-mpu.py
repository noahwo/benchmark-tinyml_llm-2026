#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single video file using a TFLite SSD model,
draws bounding boxes and labels on detected objects, and writes the annotated video
to the specified output path. It also computes and overlays a running "mAP (proxy)"
metric due to the absence of ground-truth annotations.

Phases implemented as per Programming Guidelines:
- Phase 1: Setup (imports, paths, labels, interpreter, model details)
- Phase 2: Input Acquisition & Preprocessing Loop (read frames from the given video)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling Loop
  - 4.2 Interpret Results
  - 4.3 Post-processing (confidence thresholding, coordinate scaling, clipping)
- Phase 5: Cleanup

Note on mAP:
- Since ground-truth annotations are not provided, we compute a proxy mAP by averaging
  the sorted detection scores per class (treated as a rough "AP" proxy) and then
  averaging across classes (mAP). This proxy is intended only for relative, informal
  feedback during inference and not as a formal evaluation metric.
"""

# =========================
# Phase 1: Setup
# =========================

import os
import time
import numpy as np
import cv2

# Per Programming Guideline 1.1: Import Interpreter literally as below
from ai_edge_litert.interpreter import Interpreter

# 1.2. Paths/Parameters (from CONFIGURATION PARAMETERS)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD  = 0.5

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def load_labels(label_path):
    """
    1.3. Load Labels if label path is provided.
    Returns:
        labels (list[str]): list of label strings
        label_offset (int): offset to apply for class index mapping (e.g., 1 if first label is '???')
    """
    labels = []
    if label_path and os.path.isfile(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip() != '']
    else:
        print(f"Warning: Label file not found at '{label_path}'. Proceeding with IDs as labels.")
        labels = []
    # Common in TF Lite: first label may be '???'
    label_offset = 1 if len(labels) > 0 and labels[0].strip().lower() in ('???', 'background') else 0
    return labels, label_offset

def initialize_interpreter(model_path):
    """
    1.4. Load Interpreter and allocate tensors, return input/output details.
    """
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def get_input_shape_dtype(input_details):
    """
    1.5. Extract input tensor properties (shape, dtype, index).
    Returns:
        input_index, input_height, input_width, input_channels, input_dtype, floating_model (bool)
    """
    if not input_details or len(input_details) == 0:
        raise RuntimeError("No input tensor details found in the model.")
    input_index = input_details[0]['index']
    shape = input_details[0]['shape']
    if len(shape) != 4:
        raise ValueError(f"Unexpected input tensor shape: {shape}, expected [1, H, W, C].")
    input_height, input_width, input_channels = int(shape[1]), int(shape[2]), int(shape[3])
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)
    return input_index, input_height, input_width, input_channels, input_dtype, floating_model

def parse_detection_outputs(interpreter, output_details):
    """
    4.1. Retrieve raw output tensors and attempt to parse them into standard detection outputs.
    Tries to be robust across common SSD Mobilenet TFLite detection models.

    Returns:
        boxes: np.ndarray of shape [N, 4] in order [ymin, xmin, ymax, xmax]
        classes: np.ndarray of shape [N], int32
        scores: np.ndarray of shape [N], float32
        num_detections: int
    """
    raw_outputs = []
    for od in output_details:
        raw_outputs.append(interpreter.get_tensor(od['index']))

    boxes = None
    classes = None
    scores = None
    num = None

    # Try to use names if provided
    for od, out in zip(output_details, raw_outputs):
        name = od.get('name', '').lower()
        shp = out.shape
        if out.ndim == 3 and shp[-1] == 4:
            boxes = out
        elif 'box' in name and out.ndim in (2, 3) and shp[-1] == 4:
            boxes = out
        elif 'score' in name or 'scores' in name:
            scores = out
        elif 'class' in name or 'classes' in name:
            classes = out
        elif 'num' in name:
            num = out

    # If names weren't helpful, infer by shape/value ranges
    if boxes is None:
        for out in raw_outputs:
            if out.ndim == 3 and out.shape[-1] == 4:
                boxes = out
                break
    if scores is None or classes is None:
        for out in raw_outputs:
            if out.ndim == 2:
                # Heuristic: scores are in [0,1]
                if out.size > 0 and np.amin(out) >= 0.0 and np.amax(out) <= 1.0:
                    scores = out
                else:
                    classes = out
    if num is None:
        for out in raw_outputs:
            if out.ndim == 1 and out.size == 1:
                num = out
                break

    # Squeeze to remove batch dimension if present
    if boxes is not None:
        boxes = np.squeeze(boxes)
    if classes is not None:
        classes = np.squeeze(classes)
    if scores is not None:
        scores = np.squeeze(scores)
    if num is not None:
        num_detections = int(np.squeeze(num).astype(np.int32))
    else:
        # Fallback if num_detections not provided
        num_detections = int(scores.shape[0]) if scores is not None else 0

    # Final type casting
    if classes is None:
        classes = np.zeros((num_detections,), dtype=np.int32)
    else:
        classes = classes.astype(np.int32, copy=False)
    if scores is None:
        scores = np.zeros((num_detections,), dtype=np.float32)
    else:
        scores = scores.astype(np.float32, copy=False)

    # Ensure boxes shape [N, 4]
    if boxes is None:
        boxes = np.zeros((num_detections, 4), dtype=np.float32)
    else:
        boxes = boxes.astype(np.float32, copy=False)
        if boxes.ndim == 1 and boxes.size == 4:
            boxes = boxes.reshape((1, 4))
        if boxes.shape[0] != num_detections:
            # Align length if discrepancy
            min_len = min(boxes.shape[0], num_detections, scores.shape[0])
            boxes = boxes[:min_len]
            classes = classes[:min_len]
            scores = scores[:min_len]
            num_detections = min_len

    return boxes, classes, scores, num_detections

def scale_and_clip_box(box, frame_w, frame_h, input_w=None, input_h=None):
    """
    Scales normalized or absolute boxes to frame coordinates and clips to image bounds.
    box format: [ymin, xmin, ymax, xmax]
    """
    ymin, xmin, ymax, xmax = box
    # Detect normalization: if any coord > 1.0 -> assume absolute in input image space
    if max(abs(ymin), abs(xmin), abs(ymax), abs(xmax)) <= 1.0:
        y1 = int(round(ymin * frame_h))
        x1 = int(round(xmin * frame_w))
        y2 = int(round(ymax * frame_h))
        x2 = int(round(xmax * frame_w))
    else:
        # If absolute (likely in model input coordinates), scale if input size known
        if input_w is not None and input_h is not None and input_w > 0 and input_h > 0:
            scale_x = frame_w / float(input_w)
            scale_y = frame_h / float(input_h)
            y1 = int(round(ymin * scale_y))
            x1 = int(round(xmin * scale_x))
            y2 = int(round(ymax * scale_y))
            x2 = int(round(xmax * scale_x))
        else:
            y1 = int(round(ymin))
            x1 = int(round(xmin))
            y2 = int(round(ymax))
            x2 = int(round(xmax))

    # Clip to valid ranges
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w - 1))
    y2 = max(0, min(y2, frame_h - 1))
    return x1, y1, x2, y2

def color_for_class(class_id):
    """
    Deterministic color for a given class id.
    """
    return (int(37 * class_id) % 255, int(17 * class_id) % 255, int(29 * class_id) % 255)

def compute_map_proxy(scores_per_class):
    """
    Compute a proxy mAP without ground-truth:
    - For each class, sort detection scores descending and compute mean(score) as AP proxy.
    - mAP is the mean of per-class AP proxies over classes that have detections.

    Returns:
        mAP_proxy (float)
    """
    if not scores_per_class:
        return 0.0
    ap_values = []
    for cls_id, s_list in scores_per_class.items():
        if len(s_list) == 0:
            continue
        sorted_scores = sorted(s_list, reverse=True)
        ap_proxy = float(np.mean(sorted_scores))
        ap_values.append(ap_proxy)
    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))

def main():
    # Load labels
    labels, label_offset = load_labels(LABEL_PATH)

    # Initialize interpreter and get model details
    interpreter, input_details, output_details = initialize_interpreter(MODEL_PATH)
    input_index, in_h, in_w, in_c, input_dtype, floating_model = get_input_shape_dtype(input_details)

    # 2.1. Acquire Input Data: Open the specified video file
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {INPUT_PATH}")
        return

    # Retrieve input video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # reasonable default fallback

    # Prepare VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Accumulator for mAP(proxy)
    scores_per_class = {}  # dict: class_id -> list of scores

    frame_count = 0
    t0 = time.time()

    # =========================
    # Phase 2: Input Acquisition & Preprocessing Loop
    # =========================
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # end of video

        frame_count += 1
        frame_h, frame_w = frame_bgr.shape[:2]

        # 2.2. Preprocess Data to match model input: BGR -> RGB, resize, batch, dtype handling
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        input_data = np.expand_dims(resized_rgb, axis=0)

        # 2.3. Quantization Handling
        if floating_model:
            input_data = (input_data.astype(np.float32) - 127.5) / 127.5
        else:
            input_data = input_data.astype(input_dtype, copy=False)

        # =========================
        # Phase 3: Inference
        # =========================
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # =========================
        # Phase 4: Output Interpretation & Handling Loop
        # =========================

        # 4.1. Get Output Tensor(s)
        boxes, classes, scores, num_detections = parse_detection_outputs(interpreter, output_details)

        # 4.2. Interpret Results
        # Map class indices to labels, and prepare annotations
        annotated = frame_bgr.copy()
        detections_this_frame = 0

        # 4.3. Post-processing: thresholding, coordinate scaling, clipping
        for i in range(num_detections):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(classes[i])
            # Map label with offset if needed
            label_text = None
            mapped_index = cls_id + (-label_offset) if label_offset > 0 else cls_id
            if labels and 0 <= mapped_index < len(labels):
                label_text = labels[mapped_index]
            else:
                label_text = f"id:{cls_id}"

            # Scale and clip boxes
            x1, y1, x2, y2 = scale_and_clip_box(boxes[i], frame_w, frame_h, in_w, in_h)

            # Draw bounding box and label
            color = color_for_class(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_str = f"{label_text}: {score:.2f}"
            # Put label background for readability
            (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(0, y1 - 5)
            cv2.rectangle(annotated, (x1, y_text - th - baseline), (x1 + tw, y_text + baseline), color, -1)
            cv2.putText(annotated, label_str, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Accumulate scores per class for mAP(proxy)
            if cls_id not in scores_per_class:
                scores_per_class[cls_id] = []
            scores_per_class[cls_id].append(score)
            detections_this_frame += 1

        # Compute running mAP(proxy)
        mAP_proxy = compute_map_proxy(scores_per_class)

        # 4.4. Handle Output: Overlay mAP(proxy) and write frame
        overlay_text = f"mAP(proxy): {mAP_proxy:.3f} | Detections: {detections_this_frame}"
        cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(annotated)

        # 4.5. Loop Continuation: continues until video ends

    # =========================
    # Phase 5: Cleanup
    # =========================
    cap.release()
    writer.release()

    total_time = time.time() - t0
    final_map = compute_map_proxy(scores_per_class)
    print(f"Processed {frame_count} frames in {total_time:.2f}s ({(frame_count / total_time) if total_time > 0 else 0:.2f} FPS).")
    print(f"Final mAP(proxy): {final_map:.4f}")
    print(f"Annotated output saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()