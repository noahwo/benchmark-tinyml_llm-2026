#!/usr/bin/env python3
# Application: Object Detection via a video file
# Deployment target: Raspberry Pi 4B
#
# This script follows the provided TinyML Programming Guideline phases.
# It performs TFLite inference using ai_edge_litert on a video file input,
# draws detection rectangles with labels on each frame, writes the annotated
# video to the specified output path, and overlays a computed mAP value.
# Note: Without provided ground-truth annotations, mAP is not computable;
# this script reports mAP as 0.0 (no ground truth) and overlays it.

import os
import time
import numpy as np
import cv2

# -----------------------------
# Phase 1: Setup
# -----------------------------
# 1.1 Imports (Interpreter)
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (from configuration)
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = float('0.5')  # Confidence Threshold: '0.5'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)


def load_labels(path):
    """
    Phase 1.3 Load Labels (conditional)
    Reads label file into a list of strings.
    """
    labels = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    except FileNotFoundError:
        # If labels are missing, we still proceed, mapping unknown labels by id.
        labels = []
    return labels


def prepare_interpreter(model_path_):
    """
    Phase 1.4 Load Interpreter and allocate tensors
    Phase 1.5 Get Model Details
    Returns interpreter, input_details, output_details.
    """
    interpreter_ = Interpreter(model_path=model_path_)
    interpreter_.allocate_tensors()
    input_details_ = interpreter_.get_input_details()
    output_details_ = interpreter_.get_output_details()
    return interpreter_, input_details_, output_details_


def preprocess_frame_bgr_to_input(frame_bgr, input_shape, floating_model):
    """
    Phase 2.2 Preprocess Data
    - Convert BGR to RGB
    - Resize to model input size
    - Expand dims to [1, H, W, C]
    - Normalize if floating model
    """
    _, height, width, channels = input_shape  # NHWC
    # Convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(frame_rgb, (width, height))
    # Ensure dtype
    if floating_model:
        input_data = np.asarray(resized, dtype=np.float32)
        input_data = (input_data - 127.5) / 127.5  # normalize to [-1, 1]
    else:
        input_data = np.asarray(resized, dtype=np.uint8)
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def clip_bbox(ymin, xmin, ymax, xmax):
    """
    Phase 4.3 Post-processing helper: Clip bounding boxes to [0,1]
    """
    ymin = min(max(ymin, 0.0), 1.0)
    xmin = min(max(xmin, 0.0), 1.0)
    ymax = min(max(ymax, 0.0), 1.0)
    xmax = min(max(xmax, 0.0), 1.0)
    return ymin, xmin, ymax, xmax


def get_color_for_id(id_int):
    """
    Deterministic color generation for a given integer id.
    """
    # Simple hash-based color
    r = (37 * (id_int + 1)) % 255
    g = (17 * (id_int + 7)) % 255
    b = (29 * (id_int + 13)) % 255
    return int(b), int(g), int(r)


def map_outputs_first_time(output_arrays):
    """
    Determine indices for boxes, classes, scores, and num_detections arrays
    by inspecting output shapes and value ranges.

    Expected arrays:
      - boxes: [1, N, 4]
      - classes: [1, N]
      - scores: [1, N]
      - num_detections: [1]
    """
    idx_boxes = None
    idx_classes = None
    idx_scores = None
    idx_num = None

    # Identify boxes and num_detections by shapes
    for i, arr in enumerate(output_arrays):
        if arr.ndim == 3 and arr.shape[-1] == 4:
            idx_boxes = i
        elif arr.size == 1:
            idx_num = i

    # Identify classes vs scores among 2D outputs
    candidate_1d2 = [i for i, arr in enumerate(output_arrays) if arr.ndim == 2]
    # Heuristic: scores are in [0,1], classes typically > 1
    if len(candidate_1d2) == 2:
        a_idx, b_idx = candidate_1d2
        a = output_arrays[a_idx]
        b = output_arrays[b_idx]
        a_min, a_max = float(np.min(a)), float(np.max(a))
        b_min, b_max = float(np.min(b)), float(np.max(b))
        # If a is likely scores (0..1)
        if 0.0 <= a_min and a_max <= 1.0:
            idx_scores = a_idx
            idx_classes = b_idx
        # If b is likely scores (0..1)
        elif 0.0 <= b_min and b_max <= 1.0:
            idx_scores = b_idx
            idx_classes = a_idx
        else:
            # Fallback: assume order [boxes, classes, scores, num]
            # Find one with larger range as classes
            # We'll assign arbitrarily if not sure.
            idx_classes = a_idx
            idx_scores = b_idx

    return idx_boxes, idx_classes, idx_scores, idx_num


def draw_detections_on_frame(frame_bgr, boxes, classes, scores, num_det, labels, conf_thr):
    """
    Phase 4.2 Interpretation & Phase 4.3 Post-processing
    - Threshold by confidence
    - Scale boxes to frame size
    - Clip bounding boxes to valid ranges
    - Draw rectangles and label texts
    """
    h, w = frame_bgr.shape[:2]
    num = int(num_det)
    for i in range(num):
        score = float(scores[0, i]) if scores.ndim == 2 else float(scores[i])
        if score < conf_thr:
            continue
        class_id_raw = int(classes[0, i]) if classes.ndim == 2 else int(classes[i])
        # Map label
        if 0 <= class_id_raw < len(labels):
            label_name = labels[class_id_raw]
        else:
            label_name = f"id_{class_id_raw}"
        # Get box and clip
        if boxes.ndim == 3:
            ymin, xmin, ymax, xmax = boxes[0, i]
        else:
            ymin, xmin, ymax, xmax = boxes[i]
        ymin, xmin, ymax, xmax = clip_bbox(float(ymin), float(xmin), float(ymax), float(xmax))
        # Scale to pixel coords
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        # Ensure valid rectangle
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        # Draw rectangle
        color = get_color_for_id(class_id_raw)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        # Draw label text with background
        caption = f"{label_name}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Background rectangle
        box_x1 = x1
        box_y1 = max(0, y1 - th - baseline - 4)
        box_x2 = min(w - 1, x1 + tw + 4)
        box_y2 = y1
        cv2.rectangle(frame_bgr, (box_x1, box_y1), (box_x2, box_y2), color, thickness=-1)
        # Put text
        cv2.putText(frame_bgr, caption, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return frame_bgr


def overlay_info(frame_bgr, text, at_top=True):
    """
    Draws informational text onto the frame at top-left or bottom-left.
    """
    h, w = frame_bgr.shape[:2]
    org_y = 20 if at_top else (h - 10)
    cv2.putText(frame_bgr, text, (10, org_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 240, 10), 2, cv2.LINE_AA)


def main():
    # Load labels (Phase 1.3)
    labels = load_labels(label_path)

    # Load Interpreter and get model details (Phase 1.4 & 1.5)
    interpreter, input_details, output_details = prepare_interpreter(model_path)

    # Extract input tensor details
    input_shape = input_details[0]['shape']  # [1, height, width, channels]
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # -----------------------------
    # Phase 2: Input Acquisition & Preprocessing Loop
    # -----------------------------
    # 2.1 Acquire input data: Read a single video file from input_path
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    # Prepare output video writer
    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0  # fallback if fps is not available
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (in_width, in_height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # Pre-calculate a "mAP" value; without GT it is 0.0
    # This is a computed constant noted in the overlay.
    computed_map = 0.0
    map_note = f"mAP (no GT): {computed_map:.3f}"

    # Variables for performance statistics
    frame_count = 0
    t_inference_total = 0.0

    # Output mapping indices (detected after first inference)
    out_idx_boxes = None
    out_idx_classes = None
    out_idx_scores = None
    out_idx_num = None

    # 2.4 Loop control: process all frames from the video file
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # 2.2 Preprocess frame
        input_data = preprocess_frame_bgr_to_input(frame_bgr, input_shape, floating_model)

        # 2.3 Quantization handling already applied in preprocess based on floating_model

        # -----------------------------
        # Phase 3: Inference
        # -----------------------------
        # 3.1 Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 3.2 Run inference
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        t_inference_total += (t1 - t0)

        # -----------------------------
        # Phase 4: Output Interpretation & Handling Loop
        # -----------------------------
        # 4.1 Get output tensors
        output_arrays = [interpreter.get_tensor(od['index']) for od in output_details]

        # Determine output indices on first frame
        if (out_idx_boxes is None) or (out_idx_classes is None) or (out_idx_scores is None) or (out_idx_num is None):
            out_idx_boxes, out_idx_classes, out_idx_scores, out_idx_num = map_outputs_first_time(output_arrays)
            # Safety checks
            if out_idx_boxes is None or out_idx_scores is None or out_idx_classes is None or out_idx_num is None:
                # Attempt fallback to common order [boxes, classes, scores, num_detections]
                if len(output_arrays) >= 4:
                    out_idx_boxes = out_idx_boxes if out_idx_boxes is not None else 0
                    out_idx_classes = out_idx_classes if out_idx_classes is not None else 1
                    out_idx_scores = out_idx_scores if out_idx_scores is not None else 2
                    out_idx_num = out_idx_num if out_idx_num is not None else 3
                else:
                    raise RuntimeError("Unable to map model outputs to boxes/classes/scores/num_detections.")

        # 4.2 Interpret Results
        boxes = output_arrays[out_idx_boxes]
        classes = output_arrays[out_idx_classes]
        scores = output_arrays[out_idx_scores]
        num_detections = output_arrays[out_idx_num]
        # Ensure correct dtypes
        if classes.dtype != np.int32 and classes.dtype != np.int64:
            # SSD TFLite often returns float class indices; cast to int after usage.
            pass  # We'll cast when indexing
        # num_detections may be float or int
        num_det_val = int(num_detections[0]) if num_detections.size == 1 else int(num_detections.flatten()[0])

        # 4.3 Post-processing (confidence thresholding, scaling, clipping are handled in helper)
        frame_bgr = draw_detections_on_frame(
            frame_bgr=frame_bgr,
            boxes=boxes,
            classes=classes.astype(np.int32) if classes.dtype != np.int32 else classes,
            scores=scores,
            num_det=num_det_val,
            labels=labels,
            conf_thr=CONF_THRESHOLD
        )

        # 4.4 Handle Output: overlay mAP and FPS info, then write to file
        # Compute instantaneous FPS for display
        inf_ms = (t1 - t0) * 1000.0
        fps_inst = 1000.0 / inf_ms if inf_ms > 0 else 0.0
        overlay_info(frame_bgr, f"{map_note}", at_top=True)
        overlay_info(frame_bgr, f"Inference: {inf_ms:.1f} ms ({fps_inst:.1f} FPS)", at_top=False)
        out.write(frame_bgr)

        # 4.5 Loop continuation handled by while True; break when frames end

    # -----------------------------
    # Phase 5: Cleanup
    # -----------------------------
    cap.release()
    out.release()

    # Summary logging
    if frame_count > 0:
        avg_inf_ms = (t_inference_total / frame_count) * 1000.0
        print(f"Processed {frame_count} frames.")
        print(f"Average inference time: {avg_inf_ms:.2f} ms per frame.")
        print(f"Output video saved to: {output_path}")
        print(f"Reported mAP (no ground truth provided): {computed_map:.3f}")
    else:
        print("No frames were processed. Please check the input video file.")


if __name__ == "__main__":
    main()