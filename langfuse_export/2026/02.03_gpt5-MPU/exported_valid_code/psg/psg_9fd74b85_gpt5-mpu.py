#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single video file using a TensorFlow Lite model
via ai_edge_litert Interpreter, draws bounding boxes and labels on detected objects, and
writes the annotated result to an output video. It also computes and overlays a proxy mAP
(mean Average Precision) metric based on mean confidence per detected class (no ground truth).

Phases implemented per Programming Guidelines:
- Phase 1: Setup (imports, paths, labels, interpreter, model details)
- Phase 2: Input Acquisition & Preprocessing Loop (video frames preprocessing)
- Phase 3: Inference (invocation per frame)
- Phase 4: Output Interpretation & Handling (parsing outputs, thresholding, clipping; writing annotated video and mAP proxy)
- Phase 5: Cleanup (release resources)
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1. Imports: Import interpreter literally by specified API
from ai_edge_litert.interpreter import Interpreter  # per instruction

def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def ensure_dir_for_file(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def get_tflite_output_tensors(interpreter, output_details):
    # Retrieve output tensors in the order provided
    outputs = []
    for od in output_details:
        outputs.append(interpreter.get_tensor(od['index']))
    return outputs

def identify_detection_tensors(outputs):
    """
    Identify SSD MobileNet outputs:
    Typically:
      - boxes: [1, num_detections, 4]
      - classes: [1, num_detections]
      - scores: [1, num_detections]
      - num_detections: [1]
    This function uses heuristics to map them.
    Returns: boxes, classes, scores, num_detections
    """
    boxes = None
    classes = None
    scores = None
    num_det = None

    # Flatten outer batch dimension where applicable for easier handling
    for out in outputs:
        arr = out
        shape = arr.shape
        # num_detections: usually shape == (1,) or arr.size == 1
        if arr.size == 1:
            num_det = int(np.squeeze(arr).astype(np.int32))
            continue
        # boxes: last dim == 4
        if len(shape) == 3 and shape[-1] == 4:
            boxes = np.squeeze(arr, axis=0)
            continue
        # classes or scores: shape [1, N]
        if len(shape) == 2 and shape[0] == 1:
            vals = np.squeeze(arr, axis=0)
            # scores are in [0,1]
            maxv = float(np.max(vals)) if vals.size > 0 else 0.0
            minv = float(np.min(vals)) if vals.size > 0 else 0.0
            if 0.0 <= minv and maxv <= 1.0:
                # likely scores
                # But classes could be small ints too; if all ints and within 0..1 it's ambiguous for N=1
                # Prefer to assign scores first, then classes
                if scores is None:
                    scores = vals
                else:
                    # already assigned, so this must be classes
                    classes = vals
            else:
                classes = vals
            continue

    # Fallbacks: if num_det is None, deduce from shapes present
    if num_det is None:
        if boxes is not None:
            num_det = boxes.shape[0]
        elif scores is not None:
            num_det = scores.shape[0]
        elif classes is not None:
            num_det = classes.shape[0]
        else:
            num_det = 0

    # Ensure all are present with consistent sizes
    if boxes is None:
        boxes = np.zeros((num_det, 4), dtype=np.float32)
    if scores is None:
        scores = np.zeros((num_det,), dtype=np.float32)
    if classes is None:
        classes = np.zeros((num_det,), dtype=np.float32)

    # Clip arrays to num_det if larger
    boxes = boxes[:num_det]
    scores = scores[:num_det]
    classes = classes[:num_det]

    return boxes, classes, scores, num_det

def draw_detections(frame, detections, labels, color=(0, 255, 0), thickness=2, font_scale=0.5):
    """
    Draw bounding boxes and labels on the frame.
    detections: list of dicts with keys: x1, y1, x2, y2, score, class_id, label
    """
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        score = det['score']
        class_id = det['class_id']
        label_text = det['label']

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text = f"{label_text}: {score:.2f}"
        # Draw filled rect for text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 3)), (x1 + tw + 2, y1), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

def overlay_info(frame, map_value, frame_idx, total_frames, color=(255, 0, 0)):
    """
    Overlay proxy mAP and progress information on the frame.
    """
    h, w = frame.shape[:2]
    text1 = f"mAP(proxy): {map_value:.4f}"
    text2 = f"Frame: {frame_idx}/{total_frames if total_frames>0 else '?'}"
    cv2.putText(frame, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, text2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def compute_proxy_map(per_class_scores):
    """
    Compute a proxy mAP without ground truth:
    - For each class that has detections, compute AP_proxy = mean(confidence scores)
    - mAP_proxy = mean(AP_proxy over classes with any detections)
    Returns mAP_proxy (float). If no detections observed, returns 0.0.
    """
    if not per_class_scores:
        return 0.0
    ap_list = []
    for cls_id, scores in per_class_scores.items():
        if len(scores) > 0:
            ap_list.append(float(np.mean(scores)))
    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(ap_list))

def main():
    # 1.2. Paths/Parameters (from configuration parameters)
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')

    # 1.3. Load Labels (Conditional)
    labels = []
    if label_path and os.path.exists(label_path):
        labels = load_labels(label_path)
    else:
        labels = []  # In case labels file missing, fall back to class IDs

    # 1.4. Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5. Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assume single input tensor
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # e.g., [1, 300, 300, 3]
    input_dtype = input_details[0]['dtype']
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    floating_model = (input_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1. Acquire Input Data (read a single video file from input_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    # Retrieve input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # default fallback

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output writer
    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    # For proxy mAP calculation across the whole video
    per_class_scores = {}  # dict: class_id -> list of scores

    frame_index = 0
    start_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video
        frame_index += 1

        # 2.2. Preprocess Data
        # Convert BGR to RGB for TFLite SSD models
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

        # Prepare input tensor [1, H, W, 3]
        input_data = np.expand_dims(resized, axis=0)

        # 2.3. Quantization Handling
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # Ensure uint8 for quantized models
            input_data = np.uint8(input_data)

        # Phase 3: Inference
        # 3.1. Set Input Tensor
        interpreter.set_tensor(input_index, input_data)
        # 3.2. Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling Loop
        # 4.1. Get Output Tensors
        outputs = get_tflite_output_tensors(interpreter, output_details)

        # 4.2. Interpret Results for object detection
        boxes, classes, scores, num_det = identify_detection_tensors(outputs)

        # 4.3. Post-processing: apply confidence thresholding, coordinate scaling, and clipping
        detections = []
        for i in range(num_det):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            cls_id = int(classes[i])
            label_name = labels[cls_id] if (0 <= cls_id < len(labels)) else f"id_{cls_id}"

            # boxes are in normalized coordinates [ymin, xmin, ymax, xmax]
            y_min, x_min, y_max, x_max = boxes[i].tolist()

            # Clip to [0,1]
            y_min = max(0.0, min(1.0, y_min))
            x_min = max(0.0, min(1.0, x_min))
            y_max = max(0.0, min(1.0, y_max))
            x_max = max(0.0, min(1.0, x_max))

            # Scale to pixel coordinates
            x1 = int(round(x_min * frame_width))
            y1 = int(round(y_min * frame_height))
            x2 = int(round(x_max * frame_width))
            y2 = int(round(y_max * frame_height))

            # Ensure valid rectangle within frame bounds
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width - 1, x2))
            y2 = max(0, min(frame_height - 1, y2))

            # Discard invalid boxes where min > max after rounding/clipping
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'score': score, 'class_id': cls_id, 'label': label_name
            })

            # Accumulate scores per class for proxy mAP
            if cls_id not in per_class_scores:
                per_class_scores[cls_id] = []
            per_class_scores[cls_id].append(score)

        # Compute proxy mAP up to current frame
        map_proxy = compute_proxy_map(per_class_scores)

        # 4.4. Handle Output: draw and write to file
        annotated = frame_bgr.copy()
        draw_detections(annotated, detections, labels)
        overlay_info(annotated, map_proxy, frame_index, total_frames)
        writer.write(annotated)

        # 4.5. Loop Continuation: continue until video ends

    # Phase 5: Cleanup
    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    # Print summary
    print("Processing complete.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {frame_index}")
    print(f"Elapsed time: {elapsed:.2f}s")
    final_map = compute_proxy_map(per_class_scores)
    print(f"Final mAP (proxy, mean of per-class mean confidences): {final_map:.4f}")

if __name__ == "__main__":
    main()