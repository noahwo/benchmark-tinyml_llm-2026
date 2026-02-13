import os
import sys
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports: Interpreter and EdgeTPU delegate with fallback import paths
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
    except Exception as e:
        print("ERROR: Failed to import TFLite Interpreter. Please ensure tflite_runtime or tensorflow is installed.")
        print(f"Details: {e}")
        sys.exit(1)

# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3 Load Labels (if label path is provided and needed)
def load_labels(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Handle possible formats: either "label" or "id label"
                # Given "Useful Information", labels are in plain text lines.
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1])
                else:
                    labels.append(line)
    except Exception as e:
        print(f"WARNING: Failed to read labels from {label_file_path}. Details: {e}")
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
interpreter = None
delegate_loaded = False
delegate_error_msgs = []
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
    delegate_loaded = True
except Exception as e1:
    delegate_error_msgs.append(f"Attempt 1 with 'libedgetpu.so.1.0' failed: {e1}")
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
        delegate_loaded = True
    except Exception as e2:
        delegate_error_msgs.append(f"Attempt 2 with '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0' failed: {e2}")

if not delegate_loaded or interpreter is None:
    print("ERROR: Failed to load EdgeTPU delegate. Ensure the Coral EdgeTPU runtime is installed and accessible.")
    for msg in delegate_error_msgs:
        print("-", msg)
    sys.exit(1)

try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"ERROR: Failed to allocate tensors for the interpreter. Details: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    print("ERROR: Model has no input tensors.")
    sys.exit(1)

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # Expected [1, height, width, channels]
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Helper: Identify outputs (boxes, classes, scores, count) by shape
def identify_detection_outputs(output_details_list):
    idx_boxes = idx_classes = idx_scores = idx_count = None
    for i, od in enumerate(output_details_list):
        od_shape = od.get('shape', [])
        od_dtype = od.get('dtype', None)
        # Boxes: [1, N, 4]
        if len(od_shape) == 3 and od_shape[-1] == 4:
            idx_boxes = i
        # Count: [1] or scalar
        elif len(od_shape) == 1 and od_shape[0] == 1:
            idx_count = i
        # Classes and scores: [1, N]
        elif len(od_shape) == 2 and od_shape[0] == 1:
            # Heuristic: scores are float32, classes often float32 but representable as ints
            if od_dtype == np.float32:
                # If there are two float32 [1,N], prefer the one with name containing 'scores' if exists
                tensor_name = od.get('name', '').lower()
                if 'score' in tensor_name:
                    idx_scores = i
                elif 'class' in tensor_name:
                    idx_classes = i
                else:
                    # Fallback: assign scores first, then classes
                    if idx_scores is None:
                        idx_scores = i
                    else:
                        idx_classes = i
            else:
                # Non-float for classes (rare): assume classes
                idx_classes = i
    # Final sanity fallback if names weren't helpful
    # If both classes and scores are float and ambiguous, leave as assigned
    return idx_boxes, idx_classes, idx_scores, idx_count

out_idx_boxes, out_idx_classes, out_idx_scores, out_idx_count = identify_detection_outputs(output_details)
if None in (out_idx_boxes, out_idx_classes, out_idx_scores, out_idx_count):
    print("ERROR: Failed to identify detection output tensors (boxes, classes, scores, count).")
    sys.exit(1)

# =========================
# Utility Functions
# =========================

def preprocess_frame_bgr(frame_bgr, expected_shape, dtype):
    """
    Resize and normalize BGR frame to model input.
    expected_shape: [1, height, width, channels]
    Returns input_data ready to set into interpreter.
    """
    _, in_h, in_w, in_c = expected_shape
    # Convert BGR to RGB as most detection models expect RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)
    if dtype == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)
    return input_data

def clip_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2

def get_label_name(class_id, labels_list):
    if labels_list and 0 <= class_id < len(labels_list):
        return labels_list[class_id]
    return f"Class {class_id}"

def compute_iou(box_a, box_b):
    """
    IoU between two boxes: each box is (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def voc_ap(rec, prec):
    """
    Compute AP using PASCAL VOC method.
    rec and prec are numpy arrays.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def compute_map_approx_per_frame(detections, iou_threshold=0.5):
    """
    Approximate mAP without ground truth by self-consistent clustering:
    - For each class, cluster detections by IoU > threshold (greedy by score).
    - Treat each cluster as a single 'object'.
    - Walk detections sorted by score; the first detection hitting a cluster counts as TP, others as FP.
    - Compute AP per class and average across classes that have at least one cluster.

    detections: list of dicts with keys: 'bbox'=(x1,y1,x2,y2), 'score', 'class_id'
    Returns (mAP, per_class_AP_dict)
    """
    # Group detections by class_id
    det_by_class = {}
    for d in detections:
        cid = d['class_id']
        det_by_class.setdefault(cid, []).append(d)

    ap_list = []
    per_class_ap = {}

    for cid, dets in det_by_class.items():
        if len(dets) == 0:
            continue
        # Sort by score desc
        dets_sorted = sorted(dets, key=lambda x: x['score'], reverse=True)
        # Build clusters (unique object proxies) greedily
        clusters = []  # each cluster is represented by the first (highest score) box
        for d in dets_sorted:
            assigned = False
            for rep in clusters:
                if compute_iou(d['bbox'], rep['bbox']) >= iou_threshold:
                    assigned = True
                    break
            if not assigned:
                clusters.append(d)
        K = len(clusters)
        if K == 0:
            continue

        # Assign each detection to the "closest" cluster (by IoU) if IoU >= threshold; else to a new 'no-cluster' (-1)
        # Then compute TP/FP along the ranked list: first time a cluster is hit -> TP, else FP
        cluster_hits = set()
        tp = []
        fp = []

        # Precompute IoUs to speed up
        for d in dets_sorted:
            best_iou = 0.0
            best_c = -1
            for idx_c, rep in enumerate(clusters):
                iou = compute_iou(d['bbox'], rep['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_c = idx_c
            if best_iou >= iou_threshold and best_c >= 0:
                if best_c not in cluster_hits:
                    tp.append(1)
                    fp.append(0)
                    cluster_hits.add(best_c)
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                # Does not match any cluster (consider as FP)
                tp.append(0)
                fp.append(1)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / float(K)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float32).eps)
        ap = voc_ap(rec, prec)
        ap_list.append(ap)
        per_class_ap[cid] = ap

    if len(ap_list) == 0:
        return 0.0, per_class_ap
    return float(np.mean(ap_list)), per_class_ap

def draw_detections_on_frame(frame, detections, labels_list, threshold):
    """
    Draw bounding boxes and labels for detections with score >= threshold.
    """
    h, w = frame.shape[:2]
    for det in detections:
        score = det['score']
        if score < threshold:
            continue
        x1, y1, x2, y2 = det['bbox']
        cid = det['class_id']
        label = get_label_name(cid, labels_list)
        color = (0, 255, 0)  # Green boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}: {score:.2f}"
        # Text background for readability
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, caption, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire Input Data: open video file
if not os.path.exists(input_path):
    print(f"ERROR: Input video file not found at {input_path}")
    sys.exit(1)

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"ERROR: Failed to open input video file: {input_path}")
    sys.exit(1)

# Prepare output video writer
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or np.isnan(fps):
    fps = 30.0  # default fallback

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
if not writer.isOpened():
    print(f"ERROR: Failed to open output video for writing: {output_path}")
    cap.release()
    sys.exit(1)

# 2.2 Preprocess Data will be done per-frame using preprocess_frame_bgr
# 2.3 Quantization Handling via floating_model flag inside preprocessing
# 2.4 Loop Control: process all frames in the single video

# Performance tracking
frame_count = 0
inference_times_ms = []
map_values = []

# =========================
# Processing Loop
# =========================
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_count += 1
    frame_h, frame_w = frame_bgr.shape[:2]

    # Preprocess frame
    input_data = preprocess_frame_bgr(frame_bgr, input_shape, input_dtype)

    # =========================
    # Phase 3: Inference
    # =========================
    interpreter.set_tensor(input_index, input_data)
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    infer_time_ms = (t1 - t0) * 1000.0
    inference_times_ms.append(infer_time_ms)

    # =========================
    # Phase 4: Output Interpretation & Handling
    # =========================

    # 4.1 Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[out_idx_boxes]['index'])
    classes = interpreter.get_tensor(output_details[out_idx_classes]['index'])
    scores = interpreter.get_tensor(output_details[out_idx_scores]['index'])
    count = interpreter.get_tensor(output_details[out_idx_count]['index'])

    # Flatten outputs (expected shapes: boxes [1,N,4], classes [1,N], scores [1,N], count [1])
    if boxes.ndim == 3:
        boxes = boxes[0]
    if classes.ndim == 2:
        classes = classes[0]
    if scores.ndim == 2:
        scores = scores[0]
    if np.ndim(count) > 0:
        num = int(count.flatten()[0])
    else:
        num = int(count)

    # 4.2 Interpret Results
    # Convert normalized box coordinates to absolute pixel coordinates and assemble detections
    detections = []
    for i in range(num):
        score = float(scores[i])
        cid = int(classes[i])
        y_min, x_min, y_max, x_max = boxes[i]  # typically normalized [0,1]
        # 4.3 Post-processing: apply confidence thresholding, coordinate scaling, clipping
        # We will build all detections first; filtering for drawing uses confidence_threshold
        x1 = int(x_min * frame_w)
        y1 = int(y_min * frame_h)
        x2 = int(x_max * frame_w)
        y2 = int(y_max * frame_h)
        # Ensure x1<=x2, y1<=y2
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, frame_w, frame_h)
        det = {
            'bbox': (x1, y1, x2, y2),
            'score': score,
            'class_id': cid
        }
        detections.append(det)

    # 4.3 Post-processing continued: compute approximate mAP using all detections with a self-consistent clustering approach
    # This provides an approximate metric in absence of ground-truth annotations.
    frame_map, per_class_ap = compute_map_approx_per_frame(detections, iou_threshold=0.5)
    map_values.append(frame_map)

    # 4.4 Handle Output: draw boxes and overlay info; write to output video
    annotated = frame_bgr.copy()
    draw_detections_on_frame(annotated, detections, labels, confidence_threshold)

    avg_map = float(np.mean(map_values)) if len(map_values) > 0 else 0.0
    avg_infer_ms = float(np.mean(inference_times_ms)) if len(inference_times_ms) > 0 else 0.0
    # Overlay metrics
    info_lines = [
        f"mAP (approx, this frame): {frame_map:.3f}",
        f"mAP (approx, avg): {avg_map:.3f}",
        f"Inference: {infer_time_ms:.1f} ms (avg {avg_infer_ms:.1f} ms)",
        f"Detections (>= {confidence_threshold:.2f}): {sum(1 for d in detections if d['score'] >= confidence_threshold)}"
    ]
    y0 = 25
    for idx, text in enumerate(info_lines):
        y = y0 + idx * 20
        cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2, cv2.LINE_AA)

    writer.write(annotated)

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
writer.release()

# Final report
if frame_count > 0:
    final_avg_map = float(np.mean(map_values)) if len(map_values) > 0 else 0.0
    final_avg_infer_ms = float(np.mean(inference_times_ms)) if len(inference_times_ms) > 0 else 0.0
    print("Processing complete.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {frame_count}")
    print(f"Average inference time: {final_avg_infer_ms:.2f} ms")
    print(f"Approximate mAP over video: {final_avg_map:.3f}")
else:
    print("No frames processed from the input video.")