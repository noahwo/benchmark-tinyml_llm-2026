#!/usr/bin/env python3
"""
Application: Object Detection via a video file (Raspberry Pi 4B)
Model: SSD MobileNet V1 (TFLite)
I/O:
- Input: Read a single video file from the given input_path
- Output: Output the video file with rectangles drew on the detected objects, along with texts of labels and calculated mAP (proxy, see console note)

Notes on mAP:
- Ground truth annotations are not provided, so classical mAP (requiring TP/FP matched to GT) cannot be computed.
- This script computes a proxy "precision" per class per frame using thresholding:
  TP = detections for a class with score >= threshold
  FP = detections for a class with score  < threshold
  Precision(frame, class) = TP / (TP + FP) when at least one detection for that class exists in the frame
  Per-class precision = mean over frames (where class appears)
  mAP (proxy) = mean of per-class precisions over classes that appear in the video
- The live "mAP (proxy)" is overlaid on frames and printed at the end.
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: Import interpreter literally as specified
from ai_edge_litert.interpreter import Interpreter  # Ensure this package is available on the target

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = float('0.5')  # Confidence Threshold

def ensure_dir_for_file(path: str):
    """Ensure the parent directory for a file path exists."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def load_labels(label_file: str):
    """Load labels from a file into a list (index -> label)."""
    labels = []
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"Warning: Could not read label file '{label_file}': {e}")
    return labels

def clip_bbox(xmin, ymin, xmax, ymax, width, height):
    """Clip bounding box coordinates to image boundaries."""
    xmin = max(0, min(int(xmin), width - 1))
    xmax = max(0, min(int(xmax), width - 1))
    ymin = max(0, min(int(ymin), height - 1))
    ymax = max(0, min(int(ymax), height - 1))
    return xmin, ymin, xmax, ymax

def map_detection_outputs(output_details, outputs):
    """
    Map raw output tensors to (boxes, classes, scores, num) using names or heuristics.
    Returns a tuple of numpy arrays.
    """
    boxes = None
    classes = None
    scores = None
    num = None

    # Attempt by names first
    for i, detail in enumerate(output_details):
        name = detail.get('name', '')
        name_l = name.lower() if isinstance(name, str) else ''
        val = outputs[i]
        if 'box' in name_l:
            boxes = val
        elif 'score' in name_l or 'scores' in name_l:
            scores = val
        elif 'class' in name_l:
            classes = val
        elif 'num' in name_l:
            num = val

    # Heuristic fallback if any are None
    shapes = [np.shape(o) for o in outputs]
    # Boxes: look for ...,4 in shape
    if boxes is None:
        for o in outputs:
            shp = np.shape(o)
            if len(shp) == 3 and shp[-1] == 4:
                boxes = o
                break
    # num: shape length 1 total size 1
    if num is None:
        for o in outputs:
            if np.size(o) == 1:
                num = o
                break
    # classes and scores: both typically shape (1, N)
    # Try to identify scores by value range [0,1]
    cand_1xn = [o for o in outputs if len(np.shape(o)) == 2 and np.shape(o)[0] == 1]
    if scores is None or classes is None:
        candidate_scores = None
        candidate_classes = None
        for o in cand_1xn:
            arr = o.astype(np.float32)
            arr_flat = arr.flatten()
            if arr_flat.size == 0:
                continue
            minv, maxv = float(np.min(arr_flat)), float(np.max(arr_flat))
            # If in [0, 1] range likely scores
            if 0.0 <= minv and maxv <= 1.0:
                candidate_scores = o
            else:
                candidate_classes = o
        if scores is None and candidate_scores is not None:
            scores = candidate_scores
        if classes is None and candidate_classes is not None:
            classes = candidate_classes
    return boxes, classes, scores, num

def draw_labelled_box(frame, xmin, ymin, xmax, ymax, color, label_text):
    """Draw bounding box with a label."""
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    # Text background
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    th_box = th + baseline + 4
    y_text = max(ymin, th_box)
    cv2.rectangle(frame, (xmin, y_text - th_box), (xmin + tw + 4, y_text), color, -1)
    cv2.putText(frame, label_text, (xmin + 2, y_text - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    # 1.3 Load Labels (Conditional)
    labels = load_labels(LABEL_PATH)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Using the first input only (SSD has single input)
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # e.g., [1, 300, 300, 3]
    input_dtype = input_details[0]['dtype']

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: single video file
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {INPUT_PATH}")
        return

    # Prepare output writer
    ensure_dir_for_file(OUTPUT_PATH)
    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0  # Fallback if FPS is not detected

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_width, in_height))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Setup for stats and timing
    floating_model = (input_dtype == np.float32)  # 2.3 Quantization Handling
    target_h, target_w = int(input_shape[1]), int(input_shape[2])

    # For mAP proxy calculation
    # class_id -> list of per-frame precision values
    class_prec_history = {}
    frame_count = 0
    t0 = time.time()

    print("Info: Starting inference on video.")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH} (loaded {len(labels)} labels)")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print("Note: mAP displayed is a proxy precision due to absence of ground truth.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_draw = frame_bgr.copy()
        h, w = frame_draw.shape[:2]

        # 2.2 Preprocess Data
        # - Resize to model input
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # - Convert BGR to RGB as most TFLite models expect RGB
        input_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_data = np.expand_dims(input_rgb, axis=0)
        # Ensure dtype matches model
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # uint8 model
            input_data = np.asarray(input_data, dtype=input_dtype)

        # Phase 3: Inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling
        # 4.1 Get Output Tensors
        raw_outputs = [interpreter.get_tensor(od['index']) for od in output_details]

        # 4.2 Interpret Results
        boxes, classes, scores, num = map_detection_outputs(output_details, raw_outputs)

        # Safety checks and flattening
        if boxes is None or classes is None or scores is None or num is None:
            # If mapping failed, skip this frame gracefully
            cv2.putText(frame_draw, "Output parsing error", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(frame_draw)
            continue

        # Expected shapes: boxes (1, N, 4), classes (1, N), scores (1, N), num (1,)
        boxes = np.squeeze(boxes, axis=0)
        classes = np.squeeze(classes, axis=0)
        scores = np.squeeze(scores, axis=0)
        try:
            num_det = int(np.squeeze(num).tolist())
        except Exception:
            num_det = int(np.squeeze(num).astype(np.int32))

        # Limit arrays to num_detections length if shapes allow
        n = min(num_det, boxes.shape[0], scores.shape[0], classes.shape[0])
        boxes = boxes[:n]
        classes = classes[:n]
        scores = scores[:n]

        # 4.3 Post-processing: thresholding, coordinate scaling, clipping
        # Determine if boxes are normalized [0,1] or absolute; assume normalized if max <= 1.5
        normalized = np.max(boxes) <= 1.5

        # Prepare class-wise counts for proxy precision
        frame_class_counts = {}  # class_id -> total detections in frame (above+below threshold)
        frame_class_tp = {}      # class_id -> detections above threshold

        # Draw detections
        for i in range(n):
            score = float(scores[i])
            class_id_raw = classes[i]
            # Many TFLite models output float class IDs; cast to int
            class_id = int(class_id_raw)
            frame_class_counts[class_id] = frame_class_counts.get(class_id, 0) + 1
            if score >= CONF_THRESHOLD:
                frame_class_tp[class_id] = frame_class_tp.get(class_id, 0) + 1

                ymin, xmin, ymax, xmax = boxes[i]
                if normalized:
                    xmin_a = xmin * w
                    xmax_a = ymax * w  # careful: order is [ymin, xmin, ymax, xmax]
                    xmax_a = xmax * w
                    ymin_a = ymin * h
                    ymax_a = ymax * h
                else:
                    xmin_a = xmin
                    ymin_a = ymin
                    xmax_a = xmax
                    ymax_a = ymax

                xmin_c, ymin_c, xmax_c, ymax_c = clip_bbox(xmin_a, ymin_a, xmax_a, ymax_a, w, h)

                # Choose color based on class id
                color = (37 * (class_id + 1) % 255, 17 * (class_id + 1) % 255, 29 * (class_id + 1) % 255)

                # Label text
                if 0 <= class_id < len(labels):
                    label_name = labels[class_id]
                else:
                    label_name = f"id_{class_id}"
                label_text = f"{label_name}: {score:.2f}"

                draw_labelled_box(frame_draw, xmin_c, ymin_c, xmax_c, ymax_c, color, label_text)

        # Update proxy precision stats
        for cid, total in frame_class_counts.items():
            tp = frame_class_tp.get(cid, 0)
            precision = tp / float(total) if total > 0 else 0.0
            if cid not in class_prec_history:
                class_prec_history[cid] = []
            class_prec_history[cid].append(precision)

        # Compute mAP proxy across seen classes
        per_class_means = []
        for cid, plist in class_prec_history.items():
            if len(plist) > 0:
                per_class_means.append(float(np.mean(plist)))
        mAP_proxy = float(np.mean(per_class_means)) if len(per_class_means) > 0 else 0.0

        # 4.4 Handle Output: overlay info and write frame
        # Show mAP proxy and FPS
        elapsed = time.time() - t0
        avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
        info_text = f"mAP (proxy): {mAP_proxy:.3f} | FPS: {avg_fps:.1f}"
        cv2.putText(frame_draw, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame_draw)

        # 4.5 Loop Continuation: Next frame until input exhausted

    # Phase 5: Cleanup
    cap.release()
    writer.release()

    # Final summary
    if frame_count > 0:
        per_class_means = []
        for cid, plist in class_prec_history.items():
            if len(plist) > 0:
                per_class_means.append(float(np.mean(plist)))
        final_mAP_proxy = float(np.mean(per_class_means)) if len(per_class_means) > 0 else 0.0
    else:
        final_mAP_proxy = 0.0

    print("Inference completed.")
    print(f"Processed frames: {frame_count}")
    print(f"Final mAP (proxy): {final_mAP_proxy:.4f}")
    print(f"Result saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()