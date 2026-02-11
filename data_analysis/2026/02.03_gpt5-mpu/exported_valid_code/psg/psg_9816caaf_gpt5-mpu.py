#!/usr/bin/env python3
"""
Object Detection via a video file on Raspberry Pi 4B using a TFLite SSD model.

Phases implemented as per Programming Guideline:
- Phase 1: Setup (imports, config, labels, interpreter, model I/O details)
- Phase 2: Input Acquisition & Preprocessing Loop (read video file, preprocess frames)
- Phase 3: Inference (set input, invoke interpreter)
- Phase 4: Output Interpretation & Handling Loop (decode detections, apply threshold, clip/scale boxes, draw, compute mAP proxy, write video)
- Phase 5: Cleanup (release resources)
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports (Interpreter per guideline)
from ai_edge_litert.interpreter import Interpreter

def load_labels(label_file_path):
    """Load labels from a file into a list (one label per line)."""
    labels = []
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line != '':
                labels.append(line)
    return labels

def ensure_dir_for_file(file_path):
    """Ensure directory exists for a given file path."""
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def get_output_tensors(interpreter, output_details):
    """
    Retrieve and organize typical SSD MobileNet outputs from TFLite:
    - boxes: [1, N, 4] (ymin, xmin, ymax, xmax) normalized
    - classes: [1, N]
    - scores: [1, N]
    - num_detections: [1]
    Returns a dict with keys: 'boxes', 'classes', 'scores', 'num'
    """
    outputs = {}
    for od in output_details:
        tensor = interpreter.get_tensor(od['index'])
        shape = tensor.shape
        # Identify by shape patterns
        if len(shape) == 3 and shape[-1] == 4:
            outputs['boxes'] = tensor
        elif len(shape) == 2 and shape[0] == 1 and shape[1] > 1 and tensor.dtype in (np.float32, np.int64, np.int32):
            # Could be classes or scores; differentiate by dtype/values later if both collide
            # Temporarily store to decide
            if 'scores' not in outputs and tensor.dtype == np.float32 and np.all((tensor >= 0.0) & (tensor <= 1.0)):
                outputs['scores'] = tensor
            else:
                outputs['classes'] = tensor
        elif len(shape) == 1 and shape[0] == 1:
            outputs['num'] = tensor
    # Fallback if classes/scores detection overlapped ambiguously
    if 'scores' not in outputs or 'classes' not in outputs:
        # Re-extract by checking output_details in order; typical order is boxes, classes, scores, num
        # We'll attempt a best-effort mapping
        collected = []
        for od in output_details:
            collected.append(interpreter.get_tensor(od['index']))
        for t in collected:
            if len(t.shape) == 3 and t.shape[-1] == 4:
                outputs['boxes'] = t
        for t in collected:
            if len(t.shape) == 2 and t.shape[0] == 1 and t.dtype == np.float32 and np.all((t >= 0.0) & (t <= 1.0)):
                outputs['scores'] = t
        for t in collected:
            if len(t.shape) == 2 and t.shape[0] == 1 and t.dtype != np.float32:
                outputs['classes'] = t
        for t in collected:
            if len(t.shape) == 1 and t.shape[0] == 1:
                outputs['num'] = t
    return outputs

def draw_text_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1, padding=3):
    """Draw text with a background rectangle for readability."""
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - 2 * padding), (x + tw + 2 * padding, y + baseline + padding), bg_color, -1)
    cv2.putText(img, text, (x + padding, y - padding), font, font_scale, text_color, thickness, cv2.LINE_AA)

def map_class_to_label(class_id_value, labels):
    """
    Map raw class id from model to text label robustly.
    Handles common off-by-one between 0-based and 1-based class ids.
    """
    if labels is None or len(labels) == 0:
        return f"id_{int(class_id_value)}"

    idx = int(class_id_value)
    # If idx out of range, try idx-1 (common when labels omit background)
    if 0 <= idx < len(labels):
        return labels[idx]
    elif 0 <= (idx - 1) < len(labels):
        return labels[idx - 1]
    else:
        # Clamp as last resort
        clamped = max(0, min(len(labels) - 1, idx))
        return labels[clamped]

def main():
    # 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')

    # 1.3 Load Labels (if provided/relevant)
    labels = None
    if label_path and os.path.exists(label_path):
        labels = load_labels(label_path)
    else:
        labels = []

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Assume single input
    in_idx = input_details[0]['index']
    input_shape = input_details[0]['shape']  # e.g., [1, 300, 300, 3]
    input_height, input_width = input_shape[1], input_shape[2]
    input_dtype = input_details[0]['dtype']

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data (single video file)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    # Determine video properties; fallbacks if unavailable
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0 or np.isnan(src_fps):
        src_fps = 25.0  # default fallback
    # We'll initialize writer after reading first frame to know size
    writer = None
    ensure_dir_for_file(output_path)

    # Stats for mAP proxy across classes
    # We define a proxy mAP (no ground truth available):
    # For each class: AP_proxy ≈ frames_with_at_least_one_detection / total_detections_over_threshold
    # mAP_proxy = mean(AP_proxy across classes that appeared)
    class_total_detections = {}       # class_name -> total detections counted over threshold
    class_frames_with_detection = {}  # class_name -> number of frames with at least one detection
    total_frames_processed = 0

    # Timing (optional)
    start_time = time.time()
    frame_index = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        total_frames_processed += 1
        frame_index += 1

        # Initialize writer once we know frame size
        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to open output video writer: {output_path}")

        # 2.2 Preprocess Data: convert BGR->RGB, resize to model input, create input tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

        # Expand dims to [1, H, W, C] and set dtype
        if input_dtype == np.float32:
            input_data = resized_rgb.astype(np.float32)
        else:
            input_data = resized_rgb.astype(np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        # 2.3 Quantization Handling
        floating_model = (input_dtype == np.float32)
        if floating_model:
            # Normalize to [-1, 1] as per guideline
            input_data = (np.float32(input_data) - 127.5) / 127.5
        # For quantized models (uint8), no scaling required here.

        # Phase 3: Inference
        # 3.1 Set Input Tensor
        interpreter.set_tensor(in_idx, input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling
        # 4.1 Get Output Tensors
        outputs = get_output_tensors(interpreter, output_details)
        boxes = outputs.get('boxes', None)      # shape [1, N, 4]
        classes = outputs.get('classes', None)  # shape [1, N]
        scores = outputs.get('scores', None)    # shape [1, N]
        num = outputs.get('num', None)          # shape [1]

        if boxes is None or classes is None or scores is None:
            # If outputs cannot be parsed, write original frame and continue
            writer.write(frame_bgr)
            continue

        boxes = boxes[0]
        classes = classes[0]
        scores = scores[0]
        if num is not None:
            try:
                num_detections = int(np.squeeze(num))
                num_detections = min(num_detections, boxes.shape[0], classes.shape[0], scores.shape[0])
            except Exception:
                num_detections = min(boxes.shape[0], classes.shape[0], scores.shape[0])
        else:
            num_detections = min(boxes.shape[0], classes.shape[0], scores.shape[0])

        # 4.2 Interpret Results + 4.3 Post-processing: thresholding, scaling, clipping
        fh, fw = frame_bgr.shape[:2]
        frame_class_counts = {}  # class_name -> detections count in this frame (over threshold)
        for i in range(num_detections):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            # Extract and clip box coordinates (normalized)
            ymin, xmin, ymax, xmax = boxes[i].tolist()
            ymin = max(0.0, min(1.0, ymin))
            xmin = max(0.0, min(1.0, xmin))
            ymax = max(0.0, min(1.0, ymax))
            xmax = max(0.0, min(1.0, xmax))

            # Scale to pixel coordinates
            x1 = int(xmin * fw)
            y1 = int(ymin * fh)
            x2 = int(xmax * fw)
            y2 = int(ymax * fh)

            # Ensure valid non-negative bounds
            x1 = max(0, min(fw - 1, x1))
            y1 = max(0, min(fh - 1, y1))
            x2 = max(0, min(fw - 1, x2))
            y2 = max(0, min(fh - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            # Map class to text
            class_id = classes[i]
            label_text = map_class_to_label(class_id, labels) if labels else f"id_{int(class_id)}"

            # Update per-frame stats for mAP proxy
            frame_class_counts[label_text] = frame_class_counts.get(label_text, 0) + 1

            # 4.4 Handle Output: draw boxes and labels on frame
            color = (0, 255, 0)  # green box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            caption = f"{label_text}: {score:.2f}"
            draw_text_with_bg(frame_bgr, caption, (x1, max(10, y1 - 5)), font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1, padding=3)

        # Update global mAP proxy stats after processing detections in this frame
        for cname, count_in_frame in frame_class_counts.items():
            class_total_detections[cname] = class_total_detections.get(cname, 0) + count_in_frame
            class_frames_with_detection[cname] = class_frames_with_detection.get(cname, 0) + 1

        # Compute current mAP proxy (see definition above)
        ap_values = []
        for cname in class_total_detections.keys():
            total_det = class_total_detections.get(cname, 0)
            frames_with = class_frames_with_detection.get(cname, 0)
            if total_det > 0:
                ap_values.append(frames_with / total_det)
        mAP_proxy = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0

        # Overlay current mAP proxy on the frame
        mAP_text = f"mAP (proxy): {mAP_proxy:.3f}"
        draw_text_with_bg(frame_bgr, mAP_text, (10, 25), font_scale=0.7, text_color=(255, 255, 255), bg_color=(50, 50, 50), thickness=2, padding=4)

        # 4.4 Continue handling: write annotated frame to output video
        writer.write(frame_bgr)

        # 4.5 Loop continuation is handled by while True with video frames

    # Phase 5: Cleanup
    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - start_time

    # Final mAP proxy calculation and reporting
    final_ap_values = []
    for cname in class_total_detections.keys():
        total_det = class_total_detections.get(cname, 0)
        frames_with = class_frames_with_detection.get(cname, 0)
        if total_det > 0:
            final_ap_values.append(frames_with / total_det)
    final_mAP_proxy = float(np.mean(final_ap_values)) if len(final_ap_values) > 0 else 0.0

    # Print summary
    print("Detection completed.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {total_frames_processed}")
    print(f"Elapsed time: {elapsed:.2f} s")
    if len(final_ap_values) > 0:
        print("Per-class AP proxy:")
        for cname in sorted(class_total_detections.keys()):
            total_det = class_total_detections.get(cname, 0)
            frames_with = class_frames_with_detection.get(cname, 0)
            ap_proxy = (frames_with / total_det) if total_det > 0 else 0.0
            print(f"  {cname}: AP_proxy={ap_proxy:.4f} (frames_with={frames_with}, total_detections={total_det})")
    print(f"Final mAP (proxy, no GT): {final_mAP_proxy:.4f}")

if __name__ == "__main__":
    main()