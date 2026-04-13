import os
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Import Interpreter exactly as specified
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (from configuration)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = float('0.5')  # Convert from provided string to float

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1.3 Load Labels (if needed)
def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            lbl = line.strip()
            if lbl:
                labels.append(lbl)
    return labels

labels = load_labels(LABEL_PATH)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input tensor info
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
# Expected shape: [1, height, width, channels]
in_height, in_width = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Helper to determine output indices for boxes/classes/scores/num
def determine_output_indices(output_details):
    # Attempt to use names first
    boxes_idx = classes_idx = scores_idx = num_idx = None
    for i, d in enumerate(output_details):
        name = str(d.get('name', '')).lower()
        shape = d.get('shape', [])
        if 'box' in name:
            boxes_idx = i
        elif 'class' in name:
            classes_idx = i
        elif 'score' in name:
            scores_idx = i
        elif 'num' in name:
            num_idx = i

    # Fallback by shapes if necessary
    if boxes_idx is None:
        for i, d in enumerate(output_details):
            shape = d.get('shape', [])
            if len(shape) == 3 and shape[-1] == 4:
                boxes_idx = i
                break

    if num_idx is None:
        for i, d in enumerate(output_details):
            shape = d.get('shape', [])
            if np.prod(shape) == 1:
                num_idx = i
                break

    # We will resolve classes/scores (both [1, num]) after first inference reading ranges
    return boxes_idx, classes_idx, scores_idx, num_idx

boxes_idx, classes_idx, scores_idx, num_idx = determine_output_indices(output_details)

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire input data: Open video file
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

# Prepare VideoWriter with same resolution as input video
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or np.isnan(fps):
    fps = 30.0  # Fallback if FPS not set in the file

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_width, orig_height))
if not out_writer.isOpened():
    raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

# Helper functions for Phase 2 preprocessing
def preprocess_frame(frame_bgr):
    # Resize to model input size and convert BGR to RGB
    resized = cv2.resize(frame_bgr, (in_width, in_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)

    # 2.3 Quantization handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # If model expects uint8, ensure dtype is uint8
        input_data = np.asarray(input_data, dtype=input_dtype)
    return input_data

# =========================
# Functions for Phases 3 and 4
# =========================

def run_inference(input_data):
    # Phase 3: Inference
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

def fetch_raw_outputs():
    # 4.1 Get output tensors
    outputs = []
    for d in output_details:
        outputs.append(interpreter.get_tensor(d['index']))
    return outputs

def resolve_class_score_indices(outputs, classes_idx, scores_idx):
    # Both classes and scores likely shaped [1, num]; determine by value ranges.
    def is_scores(arr):
        # Scores are typically in [0, 1]
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        return (mn >= 0.0) and (mx <= 1.0)

    if classes_idx is None or scores_idx is None:
        # Find two outputs with shape [1, num] (float)
        candidates = []
        for i, d in enumerate(output_details):
            shape = d.get('shape', [])
            if len(shape) == 2 and shape[0] == 1:
                candidates.append(i)
        # Remove already known indices like num or boxes
        if num_idx in candidates:
            candidates.remove(num_idx)
        if boxes_idx in candidates:
            candidates.remove(boxes_idx)

        # Evaluate candidates after reading their values
        if len(candidates) >= 2:
            arr0 = outputs[candidates[0]]
            arr1 = outputs[candidates[1]]
            if is_scores(arr0):
                scores_idx = candidates[0]
                classes_idx = candidates[1]
            elif is_scores(arr1):
                scores_idx = candidates[1]
                classes_idx = candidates[0]
            else:
                # Fallback: assume first is classes, second is scores
                classes_idx = candidates[0]
                scores_idx = candidates[1]
        elif len(candidates) == 1:
            # If only one candidate, try to decide
            arr = outputs[candidates[0]]
            if is_scores(arr):
                scores_idx = candidates[0]
            else:
                classes_idx = candidates[0]
        # else leave as is (unlikely)
    return classes_idx, scores_idx

def scale_and_clip_boxes(boxes_norm, frame_w, frame_h):
    # boxes_norm expected as [N, 4] with [ymin, xmin, ymax, xmax] in [0,1]
    boxes_px = []
    for (ymin, xmin, ymax, xmax) in boxes_norm:
        x1 = int(max(0, min(frame_w - 1, xmin * frame_w)))
        y1 = int(max(0, min(frame_h - 1, ymin * frame_h)))
        x2 = int(max(0, min(frame_w - 1, xmax * frame_w)))
        y2 = int(max(0, min(frame_h - 1, ymax * frame_h)))
        # Ensure proper ordering
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        boxes_px.append((x1, y1, x2, y2))
    return boxes_px

def iou(boxA, boxB):
    # boxes are (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    interArea = inter_w * inter_h
    if interArea <= 0:
        return 0.0
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom

def match_and_update_metrics(prev_dets, curr_dets, preds_by_class, gt_counts_by_class, iou_thresh=0.5):
    """
    prev_dets: list of dicts with keys: class_id, score, box (x1,y1,x2,y2)
    curr_dets: same format (current frame)
    preds_by_class: dict[class_id] -> list of (score, is_tp)
    gt_counts_by_class: dict[class_id] -> int (accumulated pseudo ground-truth instances)
    This matches current detections to previous frame detections of the same class.
    """
    # Organize by class
    prev_by_cls = {}
    curr_by_cls = {}
    for d in prev_dets:
        prev_by_cls.setdefault(d['class_id'], []).append(d)
    for d in curr_dets:
        curr_by_cls.setdefault(d['class_id'], []).append(d)

    all_classes = set(prev_by_cls.keys()).union(set(curr_by_cls.keys()))

    for cls in all_classes:
        prev_list = prev_by_cls.get(cls, [])
        curr_list = curr_by_cls.get(cls, [])
        gt_counts_by_class[cls] = gt_counts_by_class.get(cls, 0) + len(prev_list)

        if len(curr_list) == 0:
            continue

        # Build IoU matrix between prev (GT) and curr (preds)
        if len(prev_list) == 0:
            # No GT, all current are FPs
            for d in curr_list:
                preds_by_class.setdefault(cls, []).append((d['score'], 0))
            continue

        iou_matrix = np.zeros((len(prev_list), len(curr_list)), dtype=np.float32)
        for i, gt in enumerate(prev_list):
            for j, pr in enumerate(curr_list):
                iou_matrix[i, j] = iou(gt['box'], pr['box'])

        # Greedy matching on IoU
        matched_gt = set()
        matched_pr = set()
        # Flatten matrix with indices sorted by IoU descending
        pairs = [(i, j, iou_matrix[i, j]) for i in range(iou_matrix.shape[0]) for j in range(iou_matrix.shape[1])]
        pairs.sort(key=lambda x: x[2], reverse=True)

        for i, j, v in pairs:
            if v < iou_thresh:
                break
            if i in matched_gt or j in matched_pr:
                continue
            matched_gt.add(i)
            matched_pr.add(j)

        # Assign TP/FP for predictions
        for j, pr in enumerate(curr_list):
            is_tp = 1 if j in matched_pr else 0
            preds_by_class.setdefault(cls, []).append((pr['score'], is_tp))

def compute_ap(precision, recall):
    """
    VOC-style AP computation with precision envelope.
    precision, recall are 1D numpy arrays sorted by descending score.
    """
    # Append sentinel values at both ends
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Sum over recall steps where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0.0
    for i in idx:
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap

def compute_map(preds_by_class, gt_counts_by_class):
    """
    Computes mAP over classes present in gt_counts_by_class with gt > 0.
    preds_by_class: dict[class_id] -> list of (score, is_tp)
    """
    aps = []
    for cls, preds in preds_by_class.items():
        gt = gt_counts_by_class.get(cls, 0)
        if gt <= 0:
            continue
        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: -x[0])
        tps = np.array([int(p[1] == 1) for p in preds_sorted], dtype=np.float32)
        fps = 1.0 - tps
        cum_tps = np.cumsum(tps)
        cum_fps = np.cumsum(fps)
        # Avoid division by zero
        denom = cum_tps + cum_fps
        denom[denom == 0] = 1e-12
        precision = cum_tps / denom
        recall = cum_tps / float(gt)
        ap = compute_ap(precision, recall)
        aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# =========================
# Phase 2.4 Loop control and processing
# =========================

prev_detections = []  # detections from previous frame for temporal mAP proxy
preds_by_class = {}   # class_id -> list of (score, is_tp)
gt_counts_by_class = {}  # class_id -> gt count

frame_index = 0
start_time = time.time()

while True:
    # 2.1 Acquire next frame
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # 2.2 Preprocess frame into input tensor
    input_data = preprocess_frame(frame)

    # =========================
    # Phase 3: Inference
    # =========================
    run_inference(input_data)

    # =========================
    # Phase 4: Output Interpretation & Handling
    # =========================

    # 4.1 Get raw outputs
    outputs = fetch_raw_outputs()

    # Resolve indices for classes/scores at first iteration if needed
    classes_idx, scores_idx = resolve_class_score_indices(outputs, classes_idx, scores_idx)

    # Extract outputs
    boxes = outputs[boxes_idx]
    # boxes shape expected [1, num, 4]
    boxes = np.squeeze(boxes, axis=0)

    # num detections
    if num_idx is not None:
        num_raw = outputs[num_idx]
        num = int(np.squeeze(num_raw).astype(np.int32))
    else:
        # Fallback: use length of scores or boxes
        num = boxes.shape[0]

    # Classes and scores
    if classes_idx is not None:
        classes = outputs[classes_idx]
        classes = np.squeeze(classes, axis=0).astype(np.int32)
    else:
        classes = np.zeros((num,), dtype=np.int32)

    if scores_idx is not None:
        scores = outputs[scores_idx]
        scores = np.squeeze(scores, axis=0).astype(np.float32)
    else:
        # If no scores available, set to 1.0 for all
        scores = np.ones((num,), dtype=np.float32)

    # Cap arrays to num
    if boxes.shape[0] > num:
        boxes = boxes[:num]
    if classes.shape[0] > num:
        classes = classes[:num]
    if scores.shape[0] > num:
        scores = scores[:num]

    # 4.2 Interpret Results: thresholding, label mapping
    # 4.3 Post-processing: confidence filtering, coordinate scaling, clipping
    mask = scores >= CONF_THRESHOLD
    filtered_boxes_norm = boxes[mask]
    filtered_scores = scores[mask]
    filtered_classes = classes[mask].astype(int)

    # Scale normalized boxes to original frame size
    filtered_boxes_px = scale_and_clip_boxes(filtered_boxes_norm, frame_w, frame_h)

    # Build current detections list for temporal matching
    current_detections = []
    for b, s, c in zip(filtered_boxes_px, filtered_scores, filtered_classes):
        current_detections.append({'class_id': int(c), 'score': float(s), 'box': b})

    # Update temporal proxy metrics (self-consistency across frames)
    if frame_index > 0:
        match_and_update_metrics(prev_detections, current_detections, preds_by_class, gt_counts_by_class, iou_thresh=0.5)

    # Compute running mAP
    running_map = compute_map(preds_by_class, gt_counts_by_class)

    # 4.4 Handle Output: draw detections and running mAP, then write frame
    # Draw detections
    for det in current_detections:
        x1, y1, x2, y2 = det['box']
        cls_id = det['class_id']
        score = det['score']
        color = (0, 255, 0)  # Green boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # Prepare label string
        if 0 <= cls_id < len(labels):
            label_name = labels[cls_id]
        else:
            label_name = f"id_{cls_id}"
        label_text = f"{label_name}: {score:.2f}"
        # Draw label background for readability
        ((tw, th), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx1, ty1 = x1, max(0, y1 - th - 4)
        tx2, ty2 = x1 + tw + 4, y1
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, thickness=-1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay running mAP (temporal proxy)
    map_text = f"mAP (temporal proxy): {running_map:.3f}"
    cv2.putText(frame, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 230), 2, cv2.LINE_AA)

    # Write frame to output
    out_writer.write(frame)

    # 4.5 Loop continuation
    prev_detections = current_detections
    frame_index += 1

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
out_writer.release()

# Final report
final_map = compute_map(preds_by_class, gt_counts_by_class)
elapsed = time.time() - start_time
print(f"Processing completed.")
print(f"Frames processed: {frame_index}")
print(f"Elapsed time: {elapsed:.2f} sec, Avg FPS: {frame_index / elapsed if elapsed > 0 else 0:.2f}")
print(f"Final mAP (temporal proxy): {final_map:.4f}")
print(f"Output saved to: {OUTPUT_PATH}")