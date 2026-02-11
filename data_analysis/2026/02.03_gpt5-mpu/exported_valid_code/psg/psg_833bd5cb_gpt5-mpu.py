import os
import time
import numpy as np
import cv2

"""
Application: Object Detection via a video file
Target Device: Raspberry Pi 4B

This script performs object detection using a TFLite SSD MobileNet model on a single input video.
It outputs a video with bounding boxes, labels, and a running mAP (proxy) overlay. The mAP proxy is
computed as the mean of per-class average confidence scores accumulated over processed frames.

Phases implemented according to the provided Programming Guidelines:
- Phase 1: Setup (imports, paths, labels, interpreter, model I/O details)
- Phase 2: Input Acquisition & Preprocessing Loop (video read and preprocessing)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling (including 4.2 interpretation and 4.3 post-processing)
- Phase 5: Cleanup
"""

# ---------------------------
# Phase 1: Setup
# ---------------------------

# 1.1 Import interpreter literally as specified
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (provided configuration)
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1.3 Load labels (conditional)
def load_labels(path):
    labels = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    labels.append(line)
    except Exception as e:
        print(f"Warning: Failed to read labels from {path}. Error: {e}")
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Expecting a single input tensor
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# ---------------------------
# Helper Functions
# ---------------------------

def preprocess_frame(frame, in_shape, floating):
    """
    Resize and normalize frame to match model input requirements.
    """
    # Model expects [1, h, w, 3]
    _, in_h, in_w, in_c = in_shape
    # Convert BGR (OpenCV) to RGB for typical TFLite SSD models
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    if floating:
        # Convert to float32 and normalize to [-1, 1] as per guideline
        input_data = (np.float32(resized) - 127.5) / 127.5
    else:
        input_data = resized.astype(input_dtype)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def get_tflite_detection_outputs(interpreter, output_details):
    """
    Retrieve detection outputs from common SSD TFLite models.
    Returns boxes, classes, scores, num_detections
    boxes: [num, 4] in [ymin, xmin, ymax, xmax] normalized (0..1)
    classes: [num] float class indices
    scores: [num] float confidence
    num_detections: int
    """
    # Typical order: boxes, classes, scores, num_detections
    # But we will detect by shape semantics
    tensors = []
    for od in output_details:
        tensors.append(interpreter.get_tensor(od['index']))

    # Flatten possible batch dimension
    boxes = None
    classes = None
    scores = None
    num = None
    for t in tensors:
        arr = np.squeeze(t)
        if arr.ndim == 2 and arr.shape[1] == 4:
            boxes = arr
        elif arr.ndim == 1 and arr.size > 10 and arr.dtype != np.int32 and np.issubdtype(arr.dtype, np.floating):
            # Likely classes or scores; classes often floats; scores floats too. Distinguish by range.
            # heuristic: scores in [0,1], classes typically >=0 and not bounded by 1.
            if np.max(arr) <= 1.0 + 1e-6:
                scores = arr
            else:
                classes = arr
        elif arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
            num = int(np.round(float(arr)))

    # Fallbacks in case shape extraction is slightly different
    if boxes is None:
        # try to find by shape in raw tensors
        for t in tensors:
            s = np.squeeze(t)
            if s.ndim == 2 and s.shape[1] == 4:
                boxes = s
                break
    if classes is None:
        for t in tensors:
            s = np.squeeze(t)
            if s.ndim == 1 and np.max(s) > 1.0:
                classes = s
                break
    if scores is None:
        for t in tensors:
            s = np.squeeze(t)
            if s.ndim == 1 and np.max(s) <= 1.0 + 1e-6:
                scores = s
                break
    if num is None:
        # If num not provided, infer from boxes length
        num = boxes.shape[0] if boxes is not None else 0

    # Clip lengths to num
    if boxes is not None:
        boxes = boxes[:num]
    if classes is not None:
        classes = classes[:num]
    if scores is not None:
        scores = scores[:num]

    return boxes, classes, scores, num

def clip_box(xmin, ymin, xmax, ymax, width, height):
    xmin = max(0, min(xmin, width - 1))
    xmax = max(0, min(xmax, width - 1))
    ymin = max(0, min(ymin, height - 1))
    ymax = max(0, min(ymax, height - 1))
    return int(xmin), int(ymin), int(xmax), int(ymax)

def iou(box_a, box_b):
    """
    Compute IoU between two boxes in (xmin, ymin, xmax, ymax)
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1 + 1) * max(0, ya2 - ya1 + 1)
    area_b = max(0, xb2 - xb1 + 1) * max(0, yb2 - yb1 + 1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def nms_per_class(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression per class.
    detections: list of dicts with keys: 'class_id', 'score', 'box' (xmin, ymin, xmax, ymax)
    returns list of kept detections
    """
    kept = []
    # Group by class
    by_class = {}
    for det in detections:
        by_class.setdefault(det['class_id'], []).append(det)

    for cls, dets in by_class.items():
        # Sort by score descending
        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        suppressed = [False] * len(dets_sorted)
        for i in range(len(dets_sorted)):
            if suppressed[i]:
                continue
            kept.append(dets_sorted[i])
            box_i = dets_sorted[i]['box']
            for j in range(i + 1, len(dets_sorted)):
                if suppressed[j]:
                    continue
                box_j = dets_sorted[j]['box']
                if iou(box_i, box_j) > iou_threshold:
                    suppressed[j] = True
    return kept

def class_id_to_name(class_id, labels_list):
    """
    Map class id from TFLite SSD output to a human-readable label.
    Many SSD models are 1-based indices. We'll attempt id-1 mapping.
    """
    # Ensure integer
    cid = int(class_id)
    # Try 1-based indexing
    idx_1based = cid - 1
    if labels_list and 0 <= idx_1based < len(labels_list):
        return labels_list[idx_1based]
    # Fallback to 0-based
    if labels_list and 0 <= cid < len(labels_list):
        return labels_list[cid]
    return f"id_{cid}"

# ---------------------------
# Phase 2: Input Acquisition & Preprocessing Loop
# ---------------------------

# Acquire input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open input video: {input_path}")

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 25.0  # sensible fallback
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out_writer.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open output video for writing: {output_path}")

# For mAP (proxy) computation: accumulate per-class scores
per_class_scores = {}  # class_id -> list of scores

frame_count = 0
inference_times = []

# ---------------------------
# Processing Loop (single input video processed frame-by-frame)
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 2.2 Preprocess Data
    input_data = preprocess_frame(frame, input_shape, floating_model)

    # 3.1 Set Input Tensor(s)
    interpreter.set_tensor(input_index, input_data)

    # 3.2 Run Inference
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    inference_times.append((t1 - t0) * 1000.0)  # ms

    # ---------------------------
    # Phase 4: Output Interpretation & Handling
    # ---------------------------

    # 4.1 Get Output Tensors
    boxes, classes, scores, num = get_tflite_detection_outputs(interpreter, output_details)
    if boxes is None or classes is None or scores is None:
        # If model outputs are not as expected, skip frame
        out_writer.write(frame)
        continue

    # 4.2 Interpret Results (assemble detections list)
    detections = []
    for i in range(num):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        cls_id = int(classes[i])
        # boxes are normalized [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = boxes[i]
        # 4.3 Post-processing: scale to pixel coords and clip
        x1 = int(xmin * width)
        y1 = int(ymin * height)
        x2 = int(xmax * width)
        y2 = int(ymax * height)
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, width, height)
        if x2 <= x1 or y2 <= y1:
            continue  # invalid box after clipping
        detections.append({
            'class_id': cls_id,
            'score': score,
            'box': (x1, y1, x2, y2)
        })

    # 4.3 Post-processing: Non-Maximum Suppression per class
    detections_nms = nms_per_class(detections, iou_threshold=0.5)

    # Update mAP proxy accumulators: per-class mean confidence
    for det in detections_nms:
        cid = det['class_id']
        per_class_scores.setdefault(cid, []).append(det['score'])

    # Compute running mAP proxy: mean over classes of mean(scores)
    if len(per_class_scores) > 0:
        per_class_means = [np.mean(scores_list) for scores_list in per_class_scores.values() if len(scores_list) > 0]
        running_map_proxy = float(np.mean(per_class_means)) if len(per_class_means) > 0 else 0.0
    else:
        running_map_proxy = 0.0

    # 4.4 Handle Output: draw boxes and labels, overlay running mAP proxy
    for det in detections_nms:
        x1, y1, x2, y2 = det['box']
        score = det['score']
        cid = det['class_id']
        label_name = class_id_to_name(cid, labels)
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # Label text
        text = f"{label_name}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Background for text
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 200, 0), thickness=-1)
        cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay running mAP (proxy)
    map_text = f"mAP: {running_map_proxy:.3f}"
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2, cv2.LINE_AA)

    # Optionally overlay FPS/inference time
    if len(inference_times) > 0:
        avg_inf_ms = np.mean(inference_times[-30:])  # moving average over last 30 frames
        inf_text = f"Inference: {avg_inf_ms:.1f} ms"
        cv2.putText(frame, inf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 2, cv2.LINE_AA)

    # Write frame to output video
    out_writer.write(frame)

# ---------------------------
# Phase 5: Cleanup
# ---------------------------
cap.release()
out_writer.release()

# Final overall mAP (proxy)
if len(per_class_scores) > 0:
    per_class_means = [np.mean(scores_list) for scores_list in per_class_scores.values() if len(scores_list) > 0]
    overall_map_proxy = float(np.mean(per_class_means)) if len(per_class_means) > 0 else 0.0
else:
    overall_map_proxy = 0.0

print("Processing completed.")
print(f"Frames processed: {frame_count}")
if len(inference_times) > 0:
    print(f"Average inference time: {np.mean(inference_times):.2f} ms")
print(f"Estimated mAP (proxy based on mean per-class confidence): {overall_map_proxy:.4f}")
print(f"Output video saved to: {output_path}")