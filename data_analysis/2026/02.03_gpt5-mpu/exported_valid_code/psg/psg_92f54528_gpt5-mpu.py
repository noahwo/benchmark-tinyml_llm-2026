#!/usr/bin/env python3
# Application: Object Detection via a video file
# Target Device: Raspberry Pi 4B
# Description: Reads a single video file, runs TFLite SSD object detection on each frame,
#              writes an output video with bounding boxes, labels, and a running heuristic mAP.

import os
import time
import numpy as np
import cv2

# -------------------------------
# Phase 1: Setup
# -------------------------------

# 1.1 Import Interpreter (try ai_edge_litert first as required; provide robust fallbacks)
Interpreter = None
interpreter_source = None
try:
    from ai_edge_litert.interpreter import Interpreter  # literal import per guideline
    interpreter_source = "ai_edge_litert"
except Exception:
    # Fallbacks to improve robustness on Raspberry Pi environments
    try:
        from tflite_runtime.interpreter import Interpreter  # commonly available runtime on Pi
        interpreter_source = "tflite_runtime"
    except Exception:
        try:
            # As a last resort, try the TensorFlow Lite interpreter if TensorFlow is installed
            from tensorflow.lite.python.interpreter import Interpreter
            interpreter_source = "tensorflow.lite"
        except Exception as e:
            raise ImportError(
                "Failed to import TFLite Interpreter from ai_edge_litert, tflite_runtime, or tensorflow.lite. "
                "Please ensure one of these is installed."
            ) from e

# 1.2 Paths/Parameters from CONFIGURATION PARAMETERS
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1.3 Load Labels (if provided and relevant)
def load_labels(path):
    labels = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if len(name) > 0:
                    labels.append(name)
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input characteristics
input_index = input_details[0]['index']
input_dtype = input_details[0]['dtype']
input_shape = input_details[0]['shape']
# Expecting NHWC: [1, height, width, channels]
if len(input_shape) != 4:
    raise ValueError(f"Unexpected input tensor shape: {input_shape}")
input_height, input_width = int(input_shape[1]), int(input_shape[2])
input_channels = int(input_shape[3])
if input_channels != 3:
    # Most SSD MobileNet models are 3-channel RGB; enforce here
    raise ValueError(f"Expected 3-channel input, got {input_channels}")

# -------------------------------
# Helper Functions
# -------------------------------

def preprocess_frame(frame_bgr, target_width, target_height, floating_model):
    # Resize and convert color BGR->RGB as TFLite SSD typically expects RGB
    resized = cv2.resize(frame_bgr, (target_width, target_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if floating_model:
        # Normalize to [-1, 1] as per guideline
        input_data = (np.float32(rgb) - 127.5) / 127.5
    else:
        input_data = np.uint8(rgb)
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def find_detection_outputs(output_details_list):
    # Identify indices for boxes, classes, scores, and num_detections robustly
    boxes_idx = classes_idx = scores_idx = num_idx = None
    for i, od in enumerate(output_details_list):
        shape = od.get('shape', [])
        dtype = od.get('dtype', None)
        # Boxes: [..., 4] float32
        if dtype == np.float32 and len(shape) >= 2 and shape[-1] == 4:
            boxes_idx = i
        # Scores: float32 and 2D or 3D with last dim equal to number of detections
        elif dtype == np.float32 and len(shape) >= 2 and shape[-1] != 4:
            # Candidate for scores; will disambiguate with classes later
            if scores_idx is None:
                scores_idx = i
        # Classes: int or float, not boxes; prefer non-float32 or name includes 'classes'
        if ('classes' in od.get('name', '').lower()) or (dtype in [np.int32, np.int64]):
            classes_idx = i
        # Num detections: scalar or shape size 1
        if (len(shape) == 1 and shape[0] == 1) or (len(shape) == 0):
            # Usually float32
            if dtype in [np.float32, np.int32, np.int64]:
                num_idx = i
    # If classes_idx still None, try to infer: pick any float array similar to scores_idx
    if classes_idx is None:
        for i, od in enumerate(output_details_list):
            if i == boxes_idx or i == scores_idx or i == num_idx:
                continue
            shape = od.get('shape', [])
            if len(shape) >= 2:
                classes_idx = i
                break
    return boxes_idx, classes_idx, scores_idx, num_idx

def parse_detections(interpreter, output_details_list):
    # Extract raw outputs
    boxes_idx, classes_idx, scores_idx, num_idx = find_detection_outputs(output_details_list)

    boxes = interpreter.get_tensor(output_details_list[boxes_idx]['index'])
    classes = interpreter.get_tensor(output_details_list[classes_idx]['index'])
    scores = interpreter.get_tensor(output_details_list[scores_idx]['index'])
    if num_idx is not None:
        num_det = interpreter.get_tensor(output_details_list[num_idx]['index'])
    else:
        num_det = None

    # Squeeze to remove batch dimension
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    if num_det is not None:
        num_det = int(np.squeeze(num_det).astype(np.int32))
    else:
        # If not provided, infer from scores length
        num_det = scores.shape[0]

    # Align shapes to length N
    N = min(num_det, boxes.shape[0], scores.shape[0], classes.shape[0])
    return boxes[:N], classes[:N], scores[:N]

def clip_box(x1, y1, x2, y2, width, height):
    x1c = max(0, min(width - 1, int(round(x1))))
    y1c = max(0, min(height - 1, int(round(y1))))
    x2c = max(0, min(width - 1, int(round(x2))))
    y2c = max(0, min(height - 1, int(round(y2))))
    # Ensure valid ordering
    if x2c < x1c:
        x1c, x2c = x2c, x1c
    if y2c < y1c:
        y1c, y2c = y2c, y1c
    return x1c, y1c, x2c, y2c

def class_id_to_name(cid, labels_list):
    # Handle both 0-based and 1-based class indexing gracefully
    name = None
    if isinstance(cid, float):
        cid = int(cid)
    if labels_list:
        if 0 <= cid < len(labels_list):
            name = labels_list[cid]
        elif 0 <= cid - 1 < len(labels_list):
            name = labels_list[cid - 1]
    return name if name else f"class_{int(cid)}"

def iou(boxA, boxB):
    # Boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    areaA = max(0, (boxA[2] - boxA[0] + 1)) * max(0, (boxA[3] - boxA[1] + 1))
    areaB = max(0, (boxB[2] - boxB[0] + 1)) * max(0, (boxB[3] - boxB[1] + 1))
    union = areaA + areaB - interArea + 1e-9
    return interArea / union

def compute_map_heuristic(gt_per_class, preds_per_class, iou_thresh=0.5):
    # Heuristic mAP using "one best detection per class per frame" as pseudo ground truth.
    ap_values = []
    for cls_id in sorted(set(list(gt_per_class.keys()) + list(preds_per_class.keys()))):
        gts = gt_per_class.get(cls_id, {})
        preds = preds_per_class.get(cls_id, [])
        num_gt = len(gts)
        if num_gt == 0 or len(preds) == 0:
            # No meaningful AP can be computed; skip class
            continue
        # Sort predictions by descending score
        preds_sorted = sorted(preds, key=lambda x: -x['score'])
        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)
        matched_frames = set()

        for i, pred in enumerate(preds_sorted):
            f = pred['frame']
            box_p = pred['box']
            if (f in gts) and (f not in matched_frames):
                if iou(box_p, gts[f]) >= iou_thresh:
                    tp[i] = 1.0
                    matched_frames.add(f)
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (num_gt + 1e-9)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

        # VOC-style continuous interpolated AP
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = 0.0
        for i in idx:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        ap_values.append(ap)

    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))

# -------------------------------
# Phase 2: Input Acquisition & Preprocessing Loop
# -------------------------------

# 2.1 Acquire Input Data (single video file from input_path)
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Failed to open input video: {input_path}")

# Retrieve input video properties
in_fps = cap.get(cv2.CAP_PROP_FPS)
if not in_fps or in_fps <= 0.1:
    in_fps = 30.0  # fallback
in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, in_fps, (in_width, in_height))
if not out_writer.isOpened():
    raise RuntimeError(f"Failed to open output video writer at: {output_path}")

# Determine floating model
floating_model = (input_details[0]['dtype'] == np.float32)

# Structures for heuristic mAP computation
# gt_per_class: class_id -> {frame_idx: box}
# preds_per_class: class_id -> [{'score':float, 'frame':int, 'box':(x1,y1,x2,y2)}]
gt_per_class = {}
preds_per_class = {}
frame_index = 0
running_map = 0.0

# -------------------------------
# Processing Loop
# -------------------------------

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # 2.2 Preprocess Data
    input_data = preprocess_frame(frame, input_width, input_height, floating_model)

    # 2.3 Quantization Handling (handled in preprocess based on floating_model)

    # -------------------------------
    # Phase 3: Inference
    # -------------------------------
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # -------------------------------
    # Phase 4: Output Interpretation & Handling
    # -------------------------------

    # 4.1 Get Output Tensor(s)
    det_boxes, det_classes, det_scores = parse_detections(interpreter, output_details)

    # 4.2 Interpret Results
    h, w = frame.shape[:2]
    detections_to_draw = []  # list of dicts: {'box':(x1,y1,x2,y2), 'score':float, 'class_id':int, 'label':str}
    per_class_best = {}  # for heuristic GT: class_id -> {'score':float, 'box':(x1,y1,x2,y2)}

    for i in range(len(det_scores)):
        score = float(det_scores[i])
        if score < confidence_threshold:
            continue
        # TFLite SSD boxes are [ymin, xmin, ymax, xmax] in normalized coordinates
        box = det_boxes[i]
        ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        # 4.3 Post-processing: coordinate scaling and clipping
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            continue

        cls_id_raw = det_classes[i]
        if isinstance(cls_id_raw, (np.floating, float)):
            cls_id = int(cls_id_raw)
        else:
            cls_id = int(cls_id_raw)
        label_name = class_id_to_name(cls_id, labels)

        det_entry = {'box': (x1, y1, x2, y2), 'score': score, 'class_id': cls_id, 'label': label_name}
        detections_to_draw.append(det_entry)

        # Update per-class best (highest confidence) for pseudo-GT
        if cls_id not in per_class_best or score > per_class_best[cls_id]['score']:
            per_class_best[cls_id] = {'score': score, 'box': (x1, y1, x2, y2)}

        # Add all predictions to preds_per_class for heuristic mAP
        if cls_id not in preds_per_class:
            preds_per_class[cls_id] = []
        preds_per_class[cls_id].append({'score': score, 'frame': frame_index, 'box': (x1, y1, x2, y2)})

    # Update heuristic GT with best per class for current frame
    for cid, info in per_class_best.items():
        if cid not in gt_per_class:
            gt_per_class[cid] = {}
        gt_per_class[cid][frame_index] = info['box']

    # Compute running heuristic mAP over processed frames so far
    running_map = compute_map_heuristic(gt_per_class, preds_per_class, iou_thresh=0.5)

    # 4.4 Handle Output: draw bounding boxes, labels, confidences and running mAP
    for det in detections_to_draw:
        x1, y1, x2, y2 = det['box']
        label_text = f"{det['label']}: {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Text background for readability
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay running mAP
    map_text = f"mAP@0.5 (heuristic): {running_map:.3f}"
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2, cv2.LINE_AA)

    # Write frame to output
    out_writer.write(frame)

    frame_index += 1

# -------------------------------
# Phase 5: Cleanup
# -------------------------------
cap.release()
out_writer.release()
elapsed = time.time() - start_time

# Final console output
print(f"Processed {frame_index} frames in {elapsed:.2f} seconds ({(frame_index / max(elapsed,1e-6)):.2f} FPS).")
print(f"Output saved to: {output_path}")
print(f"Final heuristic mAP@0.5 over processed frames: {running_map:.4f}")
print(f"Interpreter source used: {interpreter_source}")