#!/usr/bin/env python3
# Application: Object Detection via a video file
# Target Device: Raspberry Pi 4B
# Description:
#   - Loads a TFLite SSD-Mobilenet v1 model and label map.
#   - Reads a single video file, performs object detection per frame.
#   - Draws bounding boxes with labels on detected objects.
#   - Computes and overlays a running mAP value (proxy; see note below) on the video.
#   - Writes the annotated output to a video file.
#
# Note on mAP:
#   Ground truth annotations are not provided for the input video. Therefore, classical mAP cannot be computed.
#   This implementation uses a proxy: per-class AP is approximated as the mean confidence of detections for that class,
#   and mAP is the mean of these per-class averages across all classes with at least one detection.

import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter  # Phase 1.1: Required by guideline

# -------------------------------
# Phase 1: Setup
# -------------------------------

# Phase 1.2: Configuration Parameters (provided)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = float('0.5')  # Confidence threshold

def ensure_dir_for_file(file_path: str):
    """Ensure directory for a given file path exists."""
    dir_path = os.path.dirname(os.path.abspath(file_path))
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def load_labels(label_path: str):
    """Load labels from file into a list."""
    labels = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    labels.append(line)
    except Exception as e:
        print(f"Warning: Failed to read labels from {label_path}: {e}")
    return labels

def find_output_indices(output_details):
    """
    Identify indices for boxes, classes, scores, and num_detections tensors.
    Typical SSD MobileNet v1 TFLite outputs:
      - boxes: (1, N, 4)
      - classes: (1, N)
      - scores: (1, N)
      - num_detections: (1)
    """
    idx_boxes = idx_classes = idx_scores = idx_num = None
    for i, det in enumerate(output_details):
        shape = det.get('shape', None)
        if shape is None:
            continue
        shape_tuple = tuple(shape)
        # Boxes: shape like (1, N, 4)
        if len(shape_tuple) == 3 and shape_tuple[-1] == 4:
            idx_boxes = i
        # Scores or Classes: (1, N)
        elif len(shape_tuple) == 2 and shape_tuple[0] == 1:
            # We cannot distinguish scores vs classes by shape alone; defer until we fetch data
            # We'll still record potential candidates and resolve after first inference if needed.
            pass
        # Num detections: (1) or (1,1)
        elif len(shape_tuple) in (1, 2) and np.prod(shape_tuple) == 1:
            idx_num = i
    # The conventional ordering for TFLite SSD is [boxes, classes, scores, num_detections].
    # If we got boxes and num, infer rest by position if not explicitly determined later.
    # We'll return the current best guess; will finalize after first inference if needed.
    return idx_boxes, idx_classes, idx_scores, idx_num

def label_from_class_id(class_id: int, labels: list):
    """
    Obtain label string for a class ID with robustness to indexing scheme.
    Some models use 1-based class indices while label files might be 0-based and vice versa.
    """
    name = str(class_id)
    if labels:
        # Try direct index
        if 0 <= class_id < len(labels):
            name = labels[class_id]
        # Try 1-based to 0-based mapping
        elif 0 <= (class_id - 1) < len(labels):
            name = labels[class_id - 1]
        else:
            # Fallback to best effort within bounds
            idx = max(0, min(class_id, len(labels) - 1))
            name = labels[idx]
    return name

def preprocess_frame(frame_bgr, input_details, floating_model):
    """
    Phase 2.2 & 2.3: Preprocess BGR frame to model input tensor based on input_details.
    - Resize to expected input size.
    - Convert BGR to RGB.
    - Adjust dtype; normalize if floating model: (x - 127.5)/127.5
    """
    in_info = input_details[0]
    in_shape = in_info['shape']  # Expected [1, height, width, 3]
    height, width = int(in_shape[1]), int(in_shape[2])

    # Resize and convert color
    frame_resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Prepare input tensor
    if floating_model:
        input_data = (np.float32(frame_rgb) - 127.5) / 127.5
    else:
        # For uint8 models, keep 0..255
        input_data = np.uint8(frame_rgb)

    # Expand dims to [1, H, W, 3]
    input_data = np.expand_dims(input_data, axis=0)
    # Ensure dtype matches exactly
    input_data = input_data.astype(in_info['dtype'])
    return input_data

def scale_and_clip_box(box, frame_w, frame_h):
    """
    Phase 4.3: Convert normalized ymin, xmin, ymax, xmax to pixel coords and clip to frame bounds.
    """
    ymin, xmin, ymax, xmax = box
    left = int(max(0, min(frame_w - 1, xmin * frame_w)))
    right = int(max(0, min(frame_w - 1, xmax * frame_w)))
    top = int(max(0, min(frame_h - 1, ymin * frame_h)))
    bottom = int(max(0, min(frame_h - 1, ymax * frame_h)))
    # Ensure proper ordering
    left, right = min(left, right), max(left, right)
    top, bottom = min(top, bottom), max(top, bottom)
    return left, top, right, bottom

def draw_label_with_bg(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, fg_color=(255,255,255), bg_color=(0,0,0), thickness=1, alpha=0.6, padding=2):
    """
    Draws text with a filled rectangle background for readability.
    """
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    th_total = th + baseline
    # Background rectangle
    x2 = x + tw + 2 * padding
    y2 = y + th_total + 2 * padding
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Put text
    text_org = (x + padding, y + th + padding)
    cv2.putText(img, text, text_org, font, font_scale, fg_color, thickness, cv2.LINE_AA)

def compute_proxy_map(class_conf_dict):
    """
    Compute proxy mAP:
      - For each class with >=1 detection, AP_class = mean(confidences of that class).
      - mAP = mean(AP_class over classes with detections).
    Returns mAP (float) and a dict of per-class AP.
    """
    ap_values = []
    per_class_ap = {}
    for cls_id, confs in class_conf_dict.items():
        if len(confs) > 0:
            ap = float(np.mean(confs))
            per_class_ap[cls_id] = ap
            ap_values.append(ap)
    if len(ap_values) == 0:
        return 0.0, per_class_ap
    return float(np.mean(ap_values)), per_class_ap

def main():
    # Phase 1.3: Load labels
    labels = load_labels(LABEL_PATH)

    # Phase 1.4: Load Interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Phase 1.5: Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = (input_details[0]['dtype'] == np.float32)

    # Determine output indices (best-effort pre-identification)
    idx_boxes, idx_classes, idx_scores, idx_num = find_output_indices(output_details)

    # Phase 2.1: Acquire Input Data (Open video file)
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input video not found at {INPUT_PATH}")
        return

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Failed to open input video: {INPUT_PATH}")
        return

    # Prepare output writer
    ensure_dir_for_file(OUTPUT_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # default fallback

    # Fallback for any invalid dimension
    if frame_width <= 0 or frame_height <= 0:
        # Try to read one frame to get size
        ret_tmp, frame_tmp = cap.read()
        if not ret_tmp:
            print("Error: Unable to determine video frame size.")
            cap.release()
            return
        frame_height, frame_width = frame_tmp.shape[:2]
        # Rewind capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"Error: Failed to open output writer for {OUTPUT_PATH}")
        cap.release()
        return

    # Data accumulators for proxy mAP
    class_confidences = {}  # dict: class_id -> list of confidences
    total_frames = 0
    total_infer_time = 0.0

    # -------------------------------
    # Phase 2 & 3 & 4: Processing Loop
    # -------------------------------
    while True:
        # Phase 2.1: Read next frame
        ret, frame_bgr = cap.read()
        if not ret:
            break  # Phase 4.5: Loop termination (end of video)

        total_frames += 1
        orig_h, orig_w = frame_bgr.shape[:2]

        # Phase 2.2 & 2.3: Preprocess
        input_data = preprocess_frame(frame_bgr, input_details, floating_model)

        # Phase 3.1: Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Phase 3.2: Inference
        t0 = time.time()
        interpreter.invoke()
        infer_time = time.time() - t0
        total_infer_time += infer_time

        # Phase 4.1: Retrieve output tensors
        # Fetch all outputs first
        outputs = []
        for od in output_details:
            outputs.append(interpreter.get_tensor(od['index']))

        # Resolve indices for classes and scores if unknown by analyzing shapes/dtypes
        # Typical outputs: boxes (float32), classes (float32), scores (float32), num (float32)
        # We'll detect by shape pattern and value ranges.
        boxes = classes = scores = num_det = None

        # Boxes
        if idx_boxes is not None:
            boxes = outputs[idx_boxes]
        else:
            for i, arr in enumerate(outputs):
                if arr.ndim == 3 and arr.shape[-1] == 4 and arr.shape[0] == 1:
                    boxes = arr
                    idx_boxes = i
                    break

        # num_detections
        if idx_num is not None:
            num_det = outputs[idx_num]
        else:
            for i, arr in enumerate(outputs):
                if np.prod(arr.shape) == 1:
                    num_det = arr
                    idx_num = i
                    break

        # The remaining two are classes and scores with shape (1, N)
        candidate_indices = [i for i in range(len(outputs)) if i not in [idx_boxes, idx_num] and outputs[i].ndim == 2 and outputs[i].shape[0] == 1]
        # Heuristic: scores are in [0,1], classes are integer-like IDs
        for i in candidate_indices:
            arr = outputs[i]
            arr_flat = arr.flatten()
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                # If values plausibly in [0,1]
                if arr_flat.size > 0 and np.nanmin(arr_flat) >= 0.0 and np.nanmax(arr_flat) <= 1.0:
                    scores = arr
                else:
                    classes = arr
            else:
                classes = arr
        # If ambiguous, fall back to typical ordering: 0: boxes, 1: classes, 2: scores, 3: num
        if classes is None or scores is None:
            if len(outputs) >= 4:
                boxes = boxes if boxes is not None else outputs[0]
                classes = outputs[1]
                scores = outputs[2]
                num_det = num_det if num_det is not None else outputs[3]
            else:
                # As a last resort, try to assign by ranges
                for i in candidate_indices:
                    arr = outputs[i]
                    if scores is None and (np.nanmax(arr) <= 1.0):
                        scores = arr
                    elif classes is None:
                        classes = arr

        # Squeeze outputs to remove batch dim
        if boxes is not None:
            boxes = np.squeeze(boxes, axis=0)
        if classes is not None:
            classes = np.squeeze(classes, axis=0)
        if scores is not None:
            scores = np.squeeze(scores, axis=0)
        if num_det is not None:
            num_val = int(np.squeeze(num_det).astype(np.int32))
        else:
            # If num_detections not provided, infer from boxes shape
            num_val = boxes.shape[0] if boxes is not None else 0

        # Safety checks
        if boxes is None or classes is None or scores is None:
            # Cannot process detections; write original frame and continue
            writer.write(frame_bgr)
            continue

        # Phase 4.2 & 4.3: Interpret and Post-process detections
        # Filter by confidence threshold, scale boxes to original frame size, clip, and draw
        detections_drawn = 0
        for i in range(num_val):
            score = float(scores[i])
            if score < CONF_THRESHOLD:
                continue
            cls_id_raw = classes[i]
            # Convert class id to int
            try:
                cls_id = int(cls_id_raw)
            except Exception:
                cls_id = int(np.round(float(cls_id_raw)))

            box = boxes[i]  # [ymin, xmin, ymax, xmax] normalized
            left, top, right, bottom = scale_and_clip_box(box, orig_w, orig_h)

            # Draw bounding box
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), thickness=2)

            # Prepare label text
            label_text = label_from_class_id(cls_id, labels)
            display_text = f"{label_text}: {score:.2f}"

            # Draw label background and text
            draw_label_with_bg(frame_bgr, display_text, (left, max(0, top - 20)), fg_color=(255,255,255), bg_color=(0,128,0))

            # Accumulate confidences for proxy mAP
            if cls_id not in class_confidences:
                class_confidences[cls_id] = []
            class_confidences[cls_id].append(score)
            detections_drawn += 1

        # Compute running proxy mAP and overlay
        map_value, _ = compute_proxy_map(class_confidences)
        map_text = f"mAP: {map_value:.3f} | FPS(Inf): {1.0 / infer_time:.2f}"
        draw_label_with_bg(frame_bgr, map_text, (10, 10), fg_color=(255,255,255), bg_color=(0,0,0))

        # Phase 4.4: Write annotated frame to output
        writer.write(frame_bgr)

    # -------------------------------
    # Phase 5: Cleanup
    # -------------------------------
    cap.release()
    writer.release()

    # Final stats
    if total_frames > 0:
        avg_infer_fps = total_frames / total_infer_time if total_infer_time > 0 else 0.0
    else:
        avg_infer_fps = 0.0

    final_map, per_class_ap = compute_proxy_map(class_confidences)

    print("Processing complete.")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Frames processed: {total_frames}")
    print(f"Average inference FPS: {avg_infer_fps:.2f}")
    print(f"Final mAP (proxy): {final_map:.4f}")
    if per_class_ap:
        # Print top few classes by AP for reference
        sorted_items = sorted(per_class_ap.items(), key=lambda kv: kv[1], reverse=True)
        print("Per-class AP (proxy) - top 10:")
        for cls_id, ap in sorted_items[:10]:
            name = label_from_class_id(cls_id, labels)
            print(f"  {cls_id:3d} ({name}): {ap:.4f}")

if __name__ == "__main__":
    main()