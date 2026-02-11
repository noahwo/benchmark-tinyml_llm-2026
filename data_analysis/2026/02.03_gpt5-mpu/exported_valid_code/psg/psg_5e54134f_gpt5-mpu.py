#!/usr/bin/env python3
# Application: Object Detection via a video file
# Target Device: Raspberry Pi 4B
#
# This script performs object detection on a single video file using a TFLite SSD model,
# draws bounding boxes and labels, and writes an annotated output video. It also computes
# a running "mAP (proxy)" metric (mean of per-class average confidences) since no ground
# truth annotations are provided.

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: TFLite interpreter (as requested by guideline)
from ai_edge_litert.interpreter import Interpreter  # noqa: E402


def load_labels(label_path):
    """
    Load labels from a text file; each line corresponds to a label.
    Returns a list of strings. Handles empty lines gracefully.
    """
    labels = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                lbl = line.strip()
                if lbl != '':
                    labels.append(lbl)
    except Exception as e:
        print(f"[WARN] Failed to load labels from {label_path}: {e}")
    return labels


def map_class_to_label(class_id, labels):
    """
    Map class index to human-readable label.
    If label list contains a background/??? entry as first element, use offset = 1.
    """
    if labels is None or len(labels) == 0:
        return f"id_{int(class_id)}"
    offset = 1 if labels[0].strip().lower() in ('???', 'background') else 0
    idx = int(class_id) + offset
    if 0 <= idx < len(labels):
        return labels[idx]
    # Fallback when index is out of range
    return f"id_{int(class_id)}"


def ensure_dir(path):
    """
    Ensure the directory for a given file path exists.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_ssd_outputs(interpreter, output_details):
    """
    Retrieve and parse SSD-style TFLite outputs:
    - boxes: shape [1, num, 4], normalized [ymin, xmin, ymax, xmax]
    - classes: shape [1, num]
    - scores: shape [1, num]
    - num_detections: shape [1]
    Returns (boxes, classes, scores, num) as numpy arrays (1D for classes/scores) and int for num.
    Robust mapping to handle varied tensor names and dtypes.
    """
    outputs = []
    for od in output_details:
        outputs.append(interpreter.get_tensor(od['index']))

    boxes, classes, scores, num = None, None, None, None

    # First try by name hints
    for arr, od in zip(outputs, output_details):
        name = str(od.get('name', '')).lower()
        if 'box' in name and boxes is None:
            boxes = arr
        elif 'score' in name and scores is None:
            scores = arr
        elif 'class' in name and classes is None:
            classes = arr
        elif 'num' in name and num is None:
            num = arr

    # Fallback by shape/type if any are missing
    if boxes is None:
        for arr in outputs:
            if arr.ndim == 3 and arr.shape[-1] == 4 and arr.shape[0] == 1:
                boxes = arr
                break

    # For classes and scores (both [1, N])
    if scores is None or classes is None:
        candidates = [arr for arr in outputs if arr.ndim == 2 and arr.shape[0] == 1]
        # Prefer float dtype for scores
        for arr in candidates:
            if scores is None and arr.dtype in (np.float32, np.float16):
                scores = arr
        # The other one becomes classes
        for arr in candidates:
            if not np.shares_memory(arr, scores) and classes is None:
                classes = arr

    if num is None:
        # Find a scalar output
        for arr in outputs:
            if arr.size == 1:
                num = arr
                break

    # Final shaping
    if boxes is None:
        raise RuntimeError("Could not find 'boxes' output from the model.")
    if classes is None:
        raise RuntimeError("Could not find 'classes' output from the model.")
    if scores is None:
        raise RuntimeError("Could not find 'scores' output from the model.")

    # Flatten to 1D for classes/scores; boxes -> [N,4]
    boxes = np.squeeze(boxes, axis=0) if boxes.ndim == 3 else boxes
    classes = np.squeeze(classes, axis=0) if classes.ndim == 2 else classes
    scores = np.squeeze(scores, axis=0) if scores.ndim == 2 else scores

    if num is None:
        num_dets = min(len(scores), len(classes), boxes.shape[0])
    else:
        try:
            num_val = int(np.squeeze(num).astype(np.int32))
            num_dets = min(num_val, len(scores), len(classes), boxes.shape[0])
        except Exception:
            num_dets = min(len(scores), len(classes), boxes.shape[0])

    return boxes[:num_dets], classes[:num_dets], scores[:num_dets], num_dets


def clip_bbox(xmin, ymin, xmax, ymax, width, height):
    """
    Clip bounding box coordinates to image boundaries and convert to ints.
    """
    xmin = int(max(0, min(width - 1, xmin)))
    ymin = int(max(0, min(height - 1, ymin)))
    xmax = int(max(0, min(width - 1, xmax)))
    ymax = int(max(0, min(height - 1, ymax)))
    return xmin, ymin, xmax, ymax


def compute_map_proxy(per_class_confidences):
    """
    Compute a proxy for mean Average Precision (mAP) without ground truth:
    For each class: AP_proxy = mean(confidences for that class).
    mAP_proxy = mean(AP_proxy over classes that have at least one detection).
    Returns float in [0, 1] range if confidences are in [0,1]. If no detections yet -> 0.0.
    """
    if not per_class_confidences:
        return 0.0
    ap_values = []
    for cid, confs in per_class_confidences.items():
        if confs:
            ap_values.append(float(np.mean(confs)))
    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))


def main():
    # 1.2 Paths/Parameters from configuration
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = 0.5  # float

    # Ensure output directory exists
    ensure_dir(output_path)

    # 1.3 Load labels
    labels = load_labels(label_path)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) < 1:
        raise RuntimeError("Model has no input tensor.")
    input_shape = input_details[0]['shape']  # Expect [1, height, width, 3]
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    in_dtype = input_details[0]['dtype']
    floating_model = (in_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop (Read a single video file)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback to 30 FPS if metadata is missing

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (src_w, src_h))
    if not out_writer.isOpened():
        raise RuntimeError(f"Unable to open VideoWriter for: {output_path}")

    # Metrics containers
    per_class_confidences = {}  # dict: class_id -> list of confidences
    total_frames = 0
    total_inference_time = 0.0
    total_detections = 0

    # Colors for drawing (simple palette)
    rng = np.random.default_rng(12345)
    color_cache = {}

    def get_color_for_class(cid):
        if cid not in color_cache:
            # Generate a stable color for each class id
            color_cache[cid] = tuple(int(c) for c in rng.integers(0, 255, size=3))
        return color_cache[cid]

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # End of video
            total_frames += 1

            # 2.2 Preprocess to model input size and dtype
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            input_data = np.expand_dims(resized, axis=0)

            # 2.3 Quantization Handling: normalize for floating models
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = np.uint8(input_data)

            # Phase 3: Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t0 = time.time()
            interpreter.invoke()
            inf_time = time.time() - t0
            total_inference_time += inf_time

            # Phase 4: Output Interpretation & Handling
            # 4.1 Get Output Tensors
            boxes, classes, scores, num_dets = parse_ssd_outputs(interpreter, output_details)

            # 4.2 Interpret Results: Scale boxes and map labels
            # 4.3 Post-processing: thresholding and clipping
            drawn_count = 0
            for i in range(num_dets):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue

                class_id = int(classes[i])
                label_name = map_class_to_label(class_id, labels)

                ymin, xmin, ymax, xmax = boxes[i]
                # Scale normalized coords to original frame size
                x_min = int(xmin * src_w)
                y_min = int(ymin * src_h)
                x_max = int(xmax * src_w)
                y_max = int(ymax * src_h)
                x_min, y_min, x_max, y_max = clip_bbox(x_min, y_min, x_max, y_max, src_w, src_h)

                # Draw rectangle and label on BGR frame
                color = get_color_for_class(class_id)
                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, thickness=2)

                label_text = f"{label_name}: {score:.2f}"
                # Compute box for text label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.5, min(src_w, src_h) / 800.0)
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                # Draw filled rectangle behind text for readability
                text_bg_xmin = x_min
                text_bg_ymin = max(0, y_min - text_h - baseline)
                text_bg_xmax = x_min + text_w + 2
                text_bg_ymax = y_min
                cv2.rectangle(frame_bgr, (text_bg_xmin, text_bg_ymin), (text_bg_xmax, text_bg_ymax), color, thickness=-1)
                # Put the text over it
                cv2.putText(frame_bgr, label_text, (x_min + 1, y_min - baseline - 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # Update metrics
                if class_id not in per_class_confidences:
                    per_class_confidences[class_id] = []
                per_class_confidences[class_id].append(score)
                total_detections += 1
                drawn_count += 1

            # Compute and overlay mAP (proxy) on the frame
            map_proxy = compute_map_proxy(per_class_confidences)
            map_text = f"mAP (proxy): {map_proxy:.3f} | Detections (frame): {drawn_count} | FPS: {1.0/inf_time if inf_time>0 else 0.0:.1f}"
            cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 10), 2, cv2.LINE_AA)

            # 4.4 Handle Output: write annotated frame to output video
            out_writer.write(frame_bgr)

            # 4.5 Loop continuation: continue until input video ends

    finally:
        # Phase 5: Cleanup
        cap.release()
        out_writer.release()

    # Print summary
    avg_fps = total_frames / total_inference_time if total_inference_time > 0 else 0.0
    final_map_proxy = compute_map_proxy(per_class_confidences)
    print("Processing complete.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {total_frames}")
    print(f"Total detections (>= {confidence_threshold}): {total_detections}")
    print(f"Average FPS (inference only): {avg_fps:.2f}")
    print(f"Final mAP (proxy): {final_map_proxy:.3f}")


if __name__ == "__main__":
    main()