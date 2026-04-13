#!/usr/bin/env python3
"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection on a single input video using a TFLite SSD MobileNet v1 model
and writes an annotated output video. It also computes and overlays a proxy mAP estimate based on
temporal consistency of detections across consecutive frames (no ground-truth is provided).

Phases implemented according to the Programming Guidelines:
- Phase 1: Setup (imports, paths, labels, interpreter, model details)
- Phase 2: Input Acquisition & Preprocessing Loop (video file reading and preprocessing)
- Phase 3: Inference (TFLite inference invocation)
- Phase 4: Output Interpretation & Handling Loop (post-processing, drawing, proxy mAP, writing video)
- Phase 5: Cleanup (release resources)
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Import Interpreter literally as required
from ai_edge_litert.interpreter import Interpreter  # Per guideline: must import exactly this path


def load_labels(label_path):
    """Load labels from a file into a list."""
    labels = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name != '':
                    labels.append(name)
    except Exception as e:
        print(f"[WARN] Failed to load labels from {label_path}: {e}")
    return labels


def ensure_dir_for_file(file_path):
    """Create directories needed for the given file path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def to_rgb_resized(frame_bgr, target_width, target_height):
    """Resize and convert BGR frame to RGB."""
    resized = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb


def map_class_to_label(class_id_raw, labels):
    """
    Map raw class id from TFLite output to label name.
    Handles 0-based or 1-based class indexing differences.
    """
    if not labels:
        return str(int(class_id_raw)), int(class_id_raw)

    cid = int(class_id_raw)
    # Heuristic:
    # - If cid in range [0, len(labels)-1], assume 0-based.
    # - Else if cid-1 in range, assume 1-based and convert.
    if 0 <= cid < len(labels):
        name = labels[cid]
        return name, cid
    elif 1 <= cid <= len(labels):
        name = labels[cid - 1]
        return name, cid - 1
    else:
        # Clamp into range if out of bounds
        cid_clamped = max(0, min(len(labels) - 1, cid))
        return labels[cid_clamped], cid_clamped


def find_output_indices(output_details):
    """
    Identify indices for boxes, classes, scores, and num_detections among output tensors.
    Returns a dict with keys: 'boxes', 'classes', 'scores', 'num'
    """
    idx = {'boxes': None, 'classes': None, 'scores': None, 'num': None}

    # First pass using shapes and names if available
    for i, od in enumerate(output_details):
        shape = od.get('shape', None)
        name = od.get('name', '').lower() if isinstance(od.get('name', ''), str) else ''
        if shape is not None:
            if len(shape) == 3 and shape[-1] == 4:
                idx['boxes'] = i
                continue
            if len(shape) == 1 and (shape[0] == 1 or shape[0] == 4 or shape[0] == 100):
                # Could be num_detections in some variants (usually [1])
                if 'num' in name:
                    idx['num'] = i
                    continue
            if len(shape) in (2, 3) and shape[-1] != 4:
                # candidates for classes/scores
                if 'class' in name:
                    idx['classes'] = i
                elif 'score' in name or 'confidence' in name:
                    idx['scores'] = i

    # If any still None, try heuristic by reading dtype/shape patterns later.
    return idx


def get_detection_tensors(interpreter, output_details, idx_map):
    """
    Retrieve detection outputs: boxes, classes, scores, num_detections.
    If indices were ambiguous, attempt to infer based on value ranges.
    """
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    boxes = outputs[idx_map['boxes']] if idx_map['boxes'] is not None else None
    classes = outputs[idx_map['classes']] if idx_map['classes'] is not None else None
    scores = outputs[idx_map['scores']] if idx_map['scores'] is not None else None
    num = outputs[idx_map['num']] if idx_map['num'] is not None else None

    # Try to infer missing components
    if boxes is None or classes is None or scores is None:
        # Identify boxes by last dim == 4
        if boxes is None:
            for i, arr in enumerate(outputs):
                if arr.ndim == 3 and arr.shape[-1] == 4:
                    boxes = arr
                    idx_map['boxes'] = i
                    break
        # Identify classes vs scores by value ranges:
        # - scores are floats close to [0,1]
        # - classes are ints/floats representing ids > 1
        if classes is None or scores is None:
            cand = []
            for i, arr in enumerate(outputs):
                if i == idx_map.get('boxes', -1):
                    continue
                if arr.ndim in (2, 3):
                    cand.append((i, arr))
            # Analyze candidates
            score_idx_guess = None
            class_idx_guess = None
            for i, arr in cand:
                v = arr.flatten()
                if v.size == 0:
                    continue
                vsmall = v[:min(100, v.size)]
                if np.all(vsmall >= -1.0) and np.all(vsmall <= 1.0):
                    # Likely scores (some models output [0,1])
                    score_idx_guess = i
                else:
                    # Likely classes
                    class_idx_guess = i
            if scores is None and score_idx_guess is not None:
                scores = outputs[score_idx_guess]
                idx_map['scores'] = score_idx_guess
            if classes is None and class_idx_guess is not None:
                classes = outputs[class_idx_guess]
                idx_map['classes'] = class_idx_guess

    # num detections
    if num is None:
        # Some models don't output num_detections; infer from scores shape
        if scores is not None:
            n = scores.shape[1] if scores.ndim >= 2 else scores.shape[0]
            num = np.array([n], dtype=np.float32)
        else:
            num = np.array([0], dtype=np.float32)

    return boxes, classes, scores, num


def iou_xyxy(boxA, boxB):
    """Compute IoU for two boxes in (x1, y1, x2, y2) format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = xB - xA
    inter_h = yB - yA
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def color_for_class(cid):
    """Deterministic color for a given class id."""
    # Simple hash-based color
    r = (37 * (cid + 1) % 255)
    g = (17 * (cid + 1) % 255)
    b = (97 * (cid + 1) % 255)
    return int(b), int(g), int(r)


def draw_detections(frame, detections, labels, map_proxy_value, fps_value=None):
    """Draw detections and proxy mAP on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls = det['class_id']
        score = det['score']
        label_text = det['label']
        color = color_for_class(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_text}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - 5)
        cv2.rectangle(frame, (x1, y_text - th - baseline), (x1 + tw, y_text + baseline), color, -1)
        cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    overlay_text = f"Proxy mAP: {map_proxy_value:.3f}"
    cv2.putText(frame, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 10), 2, cv2.LINE_AA)
    if fps_value is not None:
        cv2.putText(frame, f"FPS: {fps_value:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 220), 2, cv2.LINE_AA)


def compute_proxy_map(total_counts, consistent_counts):
    """
    Compute a proxy mAP using temporal consistency as a surrogate for precision.
    For each class: precision_proxy = consistent_detections / total_detections (if total > 0)
    mAP_proxy = mean over classes with total > 0
    """
    precisions = []
    for cls, total in total_counts.items():
        if total > 0:
            consistent = consistent_counts.get(cls, 0)
            precisions.append(consistent / float(total))
    if len(precisions) == 0:
        return 0.0
    return float(np.mean(precisions))


def main():
    # 1.2 Paths/Parameters from CONFIGURATION PARAMETERS
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')  # Provided as string; convert to float

    # Create output directory
    ensure_dir_for_file(output_path)

    # 1.3 Load Labels
    labels = load_labels(label_path)
    if labels:
        print(f"[INFO] Loaded {len(labels)} labels.")
    else:
        print("[WARN] No labels loaded; will use class IDs as labels.")

    # 1.4 Load Interpreter
    print(f"[INFO] Loading TFLite model from: {model_path}")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details:
        raise RuntimeError("No input details found in the model.")
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_h, input_w = int(input_shape[1]), int(input_shape[2])
    floating_model = (input_dtype == np.float32)
    print(f"[INFO] Model input shape: {input_shape}, dtype: {input_dtype}, floating_model: {floating_model}")

    # Determine output indices for detection tensors
    out_idx_map = find_output_indices(output_details)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data
    if not os.path.exists(input_path):
        print(f"[ERROR] Input video not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {input_path}")
        return

    # Prepare VideoWriter to write output annotated video
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or np.isnan(src_fps) or src_fps <= 0:
        src_fps = 30.0  # fallback
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        print(f"[ERROR] Failed to open VideoWriter for: {output_path}")
        cap.release()
        return
    print(f"[INFO] Processing video: {input_path} ({src_w}x{src_h} @ {src_fps:.2f} FPS)")
    print(f"[INFO] Writing annotated video to: {output_path}")

    # Stats for proxy mAP
    total_counts = {}       # class_id -> total detections
    consistent_counts = {}  # class_id -> detections matched temporally
    prev_detections = []    # detections from previous frame: list of dicts with bbox, class_id

    frame_index = 0
    t0 = time.time()
    last_time = t0
    smoothed_fps = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        frame_index += 1

        # 2.2 Preprocess Data
        rgb_resized = to_rgb_resized(frame_bgr, input_w, input_h)
        input_data = np.expand_dims(rgb_resized, axis=0)

        # 2.3 Quantization Handling / Normalization for floating model
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # For quantized uint8 models, use uint8 input (0..255)
            input_data = np.uint8(input_data)

        # Phase 3: Inference
        # 3.1 Set Input Tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling Loop
        # 4.1 Get Output Tensor(s)
        boxes, classes, scores, num = get_detection_tensors(interpreter, output_details, out_idx_map)

        # 4.2 Interpret Results
        # Parse detections for this frame
        detections = []
        n_det = int(num[0]) if np.ndim(num) > 0 else int(num)
        # Ensure shapes
        if boxes is None or classes is None or scores is None:
            n_det = 0

        # 4.3 Post-processing: thresholding, scaling, clipping
        for i in range(n_det):
            score = float(scores[0, i]) if scores.ndim == 2 else float(scores[i])
            if score < confidence_threshold:
                continue

            # Handle class id mapping and label lookup
            raw_class_id = classes[0, i] if classes.ndim == 2 else classes[i]
            label_text, class_id = map_class_to_label(raw_class_id, labels)

            # Boxes are [ymin, xmin, ymax, xmax] normalized
            b = boxes[0, i] if boxes.ndim == 3 else boxes[i]
            ymin, xmin, ymax, xmax = float(b[0]), float(b[1]), float(b[2]), float(b[3])

            # Scale to original frame size
            x1 = int(max(0, min(src_w - 1, xmin * src_w)))
            y1 = int(max(0, min(src_h - 1, ymin * src_h)))
            x2 = int(max(0, min(src_w - 1, xmax * src_w)))
            y2 = int(max(0, min(src_h - 1, ymax * src_h)))

            # Ensure proper box ordering after scaling/clipping
            x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
            y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class_id': class_id,
                'label': label_text,
                'score': score
            })

        # Temporal consistency-based proxy mAP updates
        # Match current detections to previous detections (same class, IoU >= 0.5)
        used_prev = set()
        for det in detections:
            cid = det['class_id']
            total_counts[cid] = total_counts.get(cid, 0) + 1

            best_iou = 0.0
            best_j = -1
            for j, pdet in enumerate(prev_detections):
                if j in used_prev:
                    continue
                if pdet['class_id'] != cid:
                    continue
                iou = iou_xyxy(det['bbox'], pdet['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.5:
                consistent_counts[cid] = consistent_counts.get(cid, 0) + 1
                used_prev.add(best_j)

        map_proxy = compute_proxy_map(total_counts, consistent_counts)

        # Estimate FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        inst_fps = (1.0 / dt) if dt > 0 else src_fps
        if smoothed_fps is None:
            smoothed_fps = inst_fps
        else:
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * inst_fps

        # 4.4 Handle Output: draw and write frame
        draw_detections(frame_bgr, detections, labels, map_proxy, fps_value=smoothed_fps)
        writer.write(frame_bgr)

        # 4.5 Loop Continuation: update previous detections
        prev_detections = detections

        # Optional: print progress every N frames
        if frame_index % 30 == 0:
            print(f"[INFO] Processed {frame_index} frames; proxy mAP: {map_proxy:.3f}; FPS: {smoothed_fps:.1f}")

    t1 = time.time()
    elapsed = t1 - t0
    print(f"[INFO] Finished processing. Frames: {frame_index}, Time: {elapsed:.2f}s, Avg FPS: {frame_index / elapsed if elapsed > 0 else 0:.1f}")
    final_map = compute_proxy_map(total_counts, consistent_counts)
    print(f"[INFO] Final proxy mAP (temporal-consistency): {final_map:.4f}")

    # Phase 5: Cleanup
    cap.release()
    writer.release()
    print(f"[INFO] Output saved to: {output_path}")


if __name__ == '__main__':
    main()