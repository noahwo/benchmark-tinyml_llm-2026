#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single input video using a TensorFlow Lite SSD model.
It draws bounding boxes with labels and confidence scores on detected objects and writes the
resulting annotated video to the specified output path.

Phases implemented per guideline:
- Phase 1: Setup (imports, paths, labels, interpreter, model details)
- Phase 2: Input Acquisition & Preprocessing Loop (read frames from video file, preprocess)
- Phase 3: Inference (set input tensor, invoke)
- Phase 4: Output Interpretation & Handling Loop
  - 4.2: Interpret raw outputs for detection
  - 4.3: Post-process detections (thresholding, scaling, clipping)
  - Write annotated frames to output video, overlay label texts and mAP status
- Phase 5: Cleanup (release resources)

Notes:
- mAP requires ground truth annotations which are not provided; thus mAP is reported as 'N/A'.
- This script uses only standard modules (os, time), numpy, cv2, and the mandated ai_edge_litert Interpreter.
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1. Imports: Interpreter from ai_edge_litert as per requirement
from ai_edge_litert.interpreter import Interpreter

def ensure_dir_exists(path):
    """Ensure the directory for the given path exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_labels(label_path):
    """Load label map file into a list of strings."""
    labels = []
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                labels.append(name)
    return labels

def map_class_to_label(class_id, labels):
    """
    Map a numeric class ID to a human-readable label.
    Handles common off-by-one between datasets/models.
    """
    if not labels:
        return f"class_{int(class_id)}"
    idx = int(class_id)
    # Prefer direct index if within range
    if 0 <= idx < len(labels):
        return labels[idx]
    # Fallback: try 1-based indexing
    if 0 <= (idx - 1) < len(labels):
        return labels[idx - 1]
    return f"class_{idx}"

def get_video_writer(output_path, width, height, fps):
    """Create a VideoWriter for mp4 output."""
    ensure_dir_exists(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer

def preprocess_frame(frame_bgr, input_shape, input_dtype, floating_model):
    """
    Preprocess frame according to model input requirements.
    - Resize to model's input spatial dimensions.
    - Convert BGR to RGB (common for TFLite SSD models).
    - Normalize if floating model as per guideline.
    """
    _, in_h, in_w, in_c = input_shape  # Expecting NHWC
    # Resize and convert color
    resized = cv2.resize(frame_bgr, (in_w, in_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if floating_model:
        input_data = (np.float32(rgb) - 127.5) / 127.5
    else:
        # For quantized models, pass as the expected dtype without normalization.
        input_data = rgb.astype(input_dtype)

    # Expand to batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    # Ensure dtype matches model
    input_data = input_data.astype(input_dtype)
    return input_data

def parse_detection_outputs(output_details, interpreter):
    """
    Retrieve and parse model outputs from the interpreter for SSD-style models.
    Returns:
      boxes: (N, 4) float32, in normalized [ymin, xmin, ymax, xmax]
      classes: (N,) int32 class IDs
      scores: (N,) float32 confidence scores
      num: int number of valid detections
    """
    # Retrieve raw outputs
    raw = []
    for od in output_details:
        raw.append(interpreter.get_tensor(od['index']))

    boxes = None
    classes = None
    scores = None
    num = None

    # Identify outputs by shape/values
    for arr in raw:
        arr_np = np.array(arr)
        if arr_np.ndim == 3 and arr_np.shape[-1] == 4:
            # Typical boxes shape: [1, N, 4]
            boxes = arr_np[0]
        elif arr_np.ndim == 2:
            # Typical classes or scores shape: [1, N]
            candidate = arr_np[0]
            # Heuristic: scores are float in [0,1]; classes are ints/floats outside [0,1]
            if np.issubdtype(candidate.dtype, np.floating):
                # Try to detect scores by range
                if np.all(candidate >= -1e-3) and np.all(candidate <= 1.0 + 1e-3):
                    scores = candidate.astype(np.float32)
                else:
                    # Could be classes represented as floats
                    classes = candidate.astype(np.int32)
            else:
                classes = candidate.astype(np.int32)
        elif arr_np.ndim == 1 and arr_np.shape[0] == 1:
            # num detections
            try:
                num = int(arr_np[0])
            except Exception:
                pass
        elif arr_np.ndim == 2 and arr_np.shape == (1, 1):
            try:
                num = int(arr_np[0, 0])
            except Exception:
                pass

    # Fallbacks if num not provided
    if num is None:
        if boxes is not None:
            num = boxes.shape[0]
        elif scores is not None:
            num = scores.shape[0]
        elif classes is not None:
            num = classes.shape[0]
        else:
            num = 0

    # Truncate arrays to num if longer
    if boxes is not None and boxes.shape[0] > num:
        boxes = boxes[:num]
    if scores is not None and scores.shape[0] > num:
        scores = scores[:num]
    if classes is not None and classes.shape[0] > num:
        classes = classes[:num]

    return boxes, classes, scores, num

def draw_label_with_background(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=0.5, text_color=(255, 255, 255),
                               bg_color=(0, 0, 0), thickness=1, padding=2):
    """
    Draw text with a filled rectangle background for readability.
    origin: bottom-left corner of the text baseline.
    """
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    # Background rectangle: top-left and bottom-right
    top_left = (x - padding, y - th - baseline - padding)
    bottom_right = (x + tw + padding, y + padding)
    cv2.rectangle(img, top_left, bottom_right, bg_color, thickness=cv2.FILLED)
    cv2.putText(img, text, (x, y - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)

def clip_bbox(xmin, ymin, xmax, ymax, width, height):
    """Clip bounding box coordinates to image bounds."""
    xmin = max(0, min(xmin, width - 1))
    xmax = max(0, min(xmax, width - 1))
    ymin = max(0, min(ymin, height - 1))
    ymax = max(0, min(ymax, height - 1))
    return int(xmin), int(ymin), int(xmax), int(ymax)

def main():
    # Configuration Parameters
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    CONF_THRESH = float('0.5')

    # Phase 1.2: Check input path
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    # Phase 1.3: Load labels if provided/needed
    labels = load_labels(label_path)

    # Phase 1.4: Load interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Phase 1.5: Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        raise RuntimeError("Expected a single input tensor for SSD model.")

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    # Acquire video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # Safe default

    writer = get_video_writer(output_path, frame_width, frame_height, fps)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for output: {output_path}")

    # mAP placeholder (no ground truth, so N/A)
    map_value_text = "N/A"

    total_frames = 0
    total_inference_time = 0.0

    # Start processing frames
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video

        total_frames += 1

        # 2.2 Preprocess
        input_data = preprocess_frame(frame_bgr, input_shape, input_dtype, floating_model)

        # 3.1 Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 3.2 Run inference
        t0 = time.time()
        interpreter.invoke()
        infer_time = (time.time() - t0) * 1000.0  # ms
        total_inference_time += infer_time

        # 4.1 Get output tensors
        boxes, classes, scores, num = parse_detection_outputs(output_details, interpreter)

        # 4.2 Interpret Results (generate human-readable detections)
        detections = []
        if num is None:
            num = 0
        for i in range(num):
            score = float(scores[i]) if scores is not None else 1.0
            if score < CONF_THRESH:
                continue
            class_id = int(classes[i]) if classes is not None else -1
            label = map_class_to_label(class_id, labels)

            if boxes is not None:
                # SSD box format: [ymin, xmin, ymax, xmax] normalized
                ymin, xmin, ymax, xmax = boxes[i]
            else:
                ymin = xmin = 0.0
                ymax = xmax = 1.0

            detections.append({
                "bbox_norm": (float(xmin), float(ymin), float(xmax), float(ymax)),
                "score": score,
                "class_id": class_id,
                "label": label
            })

        # 4.3 Post-processing: threshold already applied, now scaling and clipping
        # Draw detections on the frame
        for det in detections:
            xmin_n, ymin_n, xmax_n, ymax_n = det["bbox_norm"]

            # Scale to absolute pixel coordinates
            x1 = int(xmin_n * frame_width)
            y1 = int(ymin_n * frame_height)
            x2 = int(xmax_n * frame_width)
            y2 = int(ymax_n * frame_height)

            # Clip to image bounds
            x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, frame_width, frame_height)

            # Draw rectangle
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with score
            label_text = f"{det['label']}: {det['score']:.2f}"
            draw_label_with_background(frame_bgr, label_text, (x1, max(0, y1 - 5)),
                                       font_scale=0.5, text_color=(255, 255, 255),
                                       bg_color=(0, 128, 0), thickness=1, padding=3)

        # Overlay mAP information (N/A due to missing ground truth)
        info_text = f"mAP: {map_value_text}"
        draw_label_with_background(frame_bgr, info_text, (10, 25),
                                   font_scale=0.6, text_color=(255, 255, 255),
                                   bg_color=(50, 50, 50), thickness=1, padding=4)

        # Optional: overlay FPS (inference only)
        avg_infer_ms = infer_time
        fps_infer = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0
        fps_text = f"Infer: {avg_infer_ms:.1f} ms ({fps_infer:.1f} FPS)"
        draw_label_with_background(frame_bgr, fps_text, (10, 50),
                                   font_scale=0.6, text_color=(255, 255, 255),
                                   bg_color=(50, 50, 50), thickness=1, padding=4)

        # 4.4 Handle Output: write annotated frame to output video
        writer.write(frame_bgr)

        # 4.5 Loop Continuation: continue until frames exhausted

    # Phase 5: Cleanup
    cap.release()
    writer.release()

    # Report summary to console
    if total_frames > 0:
        avg_infer_time = total_inference_time / total_frames
        print(f"Processed {total_frames} frames.")
        print(f"Average inference time: {avg_infer_time:.2f} ms per frame.")
        print(f"Output written to: {output_path}")
        print("mAP: N/A (Ground truth annotations not provided; cannot compute mAP.)")
    else:
        print("No frames processed. Please verify the input video file.")

if __name__ == "__main__":
    main()