import os
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports: Interpreter from ai_edge_litert per guideline
from ai_edge_litert.interpreter import Interpreter

# 1.2 Paths/Parameters (from configuration)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = float('0.5')  # Confidence Threshold
PSEUDO_GT_THRESHOLD = 0.75     # Threshold to build pseudo ground-truth via high-confidence detections
IOU_THRESHOLD = 0.5            # IoU threshold for matching during AP/mAP calculation
NMS_IOU_THRESHOLD = 0.5        # IoU threshold for NMS when creating pseudo GT

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1.3 Load Labels
def load_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    else:
        # Fallback minimal labels (first 10 from provided useful info)
        labels = [
            'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light'
        ]
    return labels

labels = load_labels(LABEL_PATH)

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Store input tensor properties
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
# Expect shape [1, height, width, 3]
if len(input_shape) != 4 or input_shape[-1] != 3:
    raise RuntimeError(f'Unexpected input tensor shape: {input_shape}. Expected [1, H, W, 3].')

input_height = int(input_shape[1])
input_width = int(input_shape[2])
floating_model = (input_dtype == np.float32)

# =========================
# Utility functions
# =========================

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def normalize_input(img):
    # Normalize to [-1, 1] if floating model
    return (np.float32(img) - 127.5) / 127.5

def clip_bbox(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(0, min(xmax, w - 1))
    ymax = max(0, min(ymax, h - 1))
    return xmin, ymin, xmax, ymax

def safe_label_name(class_id, labels_list):
    # Try 0-based first
    idx = int(class_id)
    if 0 <= idx < len(labels_list):
        return labels_list[idx]
    # Then try 1-based (common in TF OD models)
    idx = int(class_id) - 1
    if 0 <= idx < len(labels_list):
        return labels_list[idx]
    return f'class_{int(class_id)}'

def color_for_class(class_id):
    # Deterministic pseudo-random color for class id
    rng = np.random.RandomState(int(class_id) + 12345)
    color = rng.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def iou(boxA, boxB):
    # boxes: (xmin, ymin, xmax, ymax)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom

def nms_boxes(boxes, scores, iou_thresh=0.5):
    # Non-Maximum Suppression, returns indices of kept boxes
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou_vals = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou_vals <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def compute_ap(recall, precision):
    # Compute AP using precision envelope method
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under PR curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

def compute_map(predictions_by_class, gt_by_class, iou_thresh=0.5):
    # predictions_by_class: {class_id: [{'frame': int, 'box':(xmin,ymin,xmax,ymax), 'score': float}, ...]}
    # gt_by_class: {class_id: {frame_id: [ (xmin,ymin,xmax,ymax), ... ]}}
    ap_list = []
    for cls_id in sorted(gt_by_class.keys()):
        # Total GT count for this class
        frame_to_gts = gt_by_class[cls_id]
        total_gts = sum(len(bxs) for bxs in frame_to_gts.values())
        if total_gts == 0:
            continue

        preds = predictions_by_class.get(cls_id, [])
        if len(preds) == 0:
            ap_list.append(0.0)
            continue

        # Sort predictions by score desc
        preds_sorted = sorted(preds, key=lambda d: d['score'], reverse=True)

        # Prepare matched flags per frame GTs
        matched = {f: np.zeros(len(frame_to_gts[f]), dtype=bool) for f in frame_to_gts.keys()}

        tp = np.zeros(len(preds_sorted))
        fp = np.zeros(len(preds_sorted))

        for i, p in enumerate(preds_sorted):
            f = p['frame']
            pbox = p['box']
            gts = frame_to_gts.get(f, [])
            best_iou = 0.0
            best_j = -1
            if len(gts) > 0:
                for j, gtbox in enumerate(gts):
                    iouv = iou(pbox, gtbox)
                    if iouv > best_iou:
                        best_iou = iouv
                        best_j = j

            if best_iou >= iou_thresh and best_j >= 0 and not matched[f][best_j]:
                tp[i] = 1
                matched[f][best_j] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / (total_gts + 1e-9)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap = compute_ap(recall, precision)
        ap_list.append(ap)

    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(ap_list))

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open input video file: {INPUT_PATH}")

# Get video properties
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or np.isnan(fps):
    fps = 25.0  # Fallback FPS

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_width, orig_height))
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open output video file for writing: {OUTPUT_PATH}")

# Storage for mAP computation across frames
predictions_by_class = {}  # {class_id: [ {'frame': int, 'box': (xmin,ymin,xmax,ymax), 'score': float}, ... ]}
gt_by_class = {}           # {class_id: {frame_id: [ (xmin,ymin,xmax,ymax), ... ]}}

frame_index = 0
start_time = time.time()

# =========================
# Main processing loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    frame_h, frame_w = frame.shape[:2]

    # 2.2 Preprocess Data
    resized = cv2.resize(frame, (input_width, input_height))
    rgb = bgr_to_rgb(resized)
    input_data = np.expand_dims(rgb, axis=0)

    # 2.3 Quantization Handling
    if floating_model:
        input_data = normalize_input(input_data)
    else:
        input_data = input_data.astype(input_dtype)

    # =========================
    # Phase 3: Inference
    # =========================
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # =========================
    # Phase 4: Output Interpretation & Handling
    # =========================

    # 4.1 Get Output Tensors
    # Typical TFLite SSD outputs: boxes [1, N, 4], classes [1, N], scores [1, N], num_detections [1]
    boxes = None
    classes_out = None
    scores = None
    num_dets = None

    # Retrieve tensors and infer which is which by shape
    for od in output_details:
        out = interpreter.get_tensor(od['index'])
        shp = out.shape
        if len(shp) == 3 and shp[-1] == 4:
            boxes = np.squeeze(out, axis=0)
        elif len(shp) == 2 and shp[0] == 1 and shp[1] >= 1 and out.dtype in (np.float32, np.int64, np.int32):
            # Could be classes or scores; differentiate by dtype and value range after we read others
            # We'll temporarily store and decide later
            if classes_out is None:
                classes_out = np.squeeze(out, axis=0)
            else:
                # Decide which is score by checking range
                if scores is None:
                    scores = np.squeeze(out, axis=0)
                else:
                    # If both already set, assign to num_dets
                    num_dets = int(np.squeeze(out))
        elif len(shp) == 2 and shp[0] == 1 and shp[1] == 4:
            # Some models may return [1, 4] boxes - unlikely for SSD
            boxes = out[0]
        elif len(shp) == 1 and shp[0] == 1:
            num_dets = int(np.squeeze(out))
        else:
            # Attempt to categorize by content
            if shp[0] == 1 and shp[-1] == 4:
                boxes = np.squeeze(out, axis=0)

    # If ambiguous mapping, attempt to swap if necessary
    # Ensure scores in [0,1]
    if scores is None or np.max(scores) > 1.0 or np.min(scores) < 0.0:
        # Try to identify which of classes_out/scores is actually scores
        # If classes are integer-like, choose other as scores
        cand1 = classes_out
        # Find another 1D output if available
        for od in output_details:
            arr = interpreter.get_tensor(od['index'])
            arr_s = np.squeeze(arr)
            if arr_s.ndim == 1 and arr_s.shape[0] == cand1.shape[0] and arr_s is not cand1:
                # Choose arr_s as scores if it looks like probabilities
                if np.max(arr_s) <= 1.0 and np.min(arr_s) >= 0.0:
                    scores = arr_s
                    break
        # If still None, fallback to zeros to avoid crash
        if scores is None:
            scores = np.zeros_like(classes_out, dtype=np.float32)

    # Ensure classes are integer array
    if classes_out is not None and classes_out.dtype != np.int32 and classes_out.dtype != np.int64:
        # Round to nearest int for class ids
        classes_int = np.rint(classes_out).astype(np.int32)
    else:
        classes_int = classes_out.astype(np.int32) if classes_out is not None else np.zeros_like(scores, dtype=np.int32)

    # Ensure boxes exist
    if boxes is None:
        # Fallback: create zero boxes to avoid crash
        boxes = np.zeros((scores.shape[0], 4), dtype=np.float32)

    # If num_dets is provided, limit arrays to actual detections
    if num_dets is not None and num_dets > 0 and num_dets <= boxes.shape[0]:
        boxes = boxes[:num_dets]
        classes_int = classes_int[:num_dets]
        scores = scores[:num_dets]

    # 4.2 Interpret Results: convert to pixel coords, map labels
    detections = []  # Per-frame detections used for drawing and predictions collection
    # boxes are expected normalized [ymin, xmin, ymax, xmax]
    for i in range(len(scores)):
        score = float(scores[i])
        if score <= 0.0:
            continue
        # Extract bbox
        bymin, bxmin, bymax, bxmax = boxes[i]
        # Clip normalized coords
        bymin = float(np.clip(bymin, 0.0, 1.0))
        bxmin = float(np.clip(bxmin, 0.0, 1.0))
        bymax = float(np.clip(bymax, 0.0, 1.0))
        bxmax = float(np.clip(bxmax, 0.0, 1.0))
        # Scale to pixel coords
        xmin = int(bxmin * frame_w)
        ymin = int(bymin * frame_h)
        xmax = int(bxmax * frame_w)
        ymax = int(bymax * frame_h)
        xmin, ymin, xmax, ymax = clip_bbox(xmin, ymin, xmax, ymax, frame_w, frame_h)
        if xmax <= xmin or ymax <= ymin:
            continue
        cls_id = int(classes_int[i])
        label_name = safe_label_name(cls_id, labels)
        detections.append({
            'class_id': cls_id,
            'label': label_name,
            'score': score,
            'box': (xmin, ymin, xmax, ymax)
        })

    # 4.3 Post-processing: thresholding and NMS for pseudo-GT; clip already handled
    # Add predictions above CONF_THRESHOLD to the collection (for mAP calc)
    for det in detections:
        if det['score'] >= CONF_THRESHOLD:
            cls = det['class_id']
            predictions_by_class.setdefault(cls, []).append({
                'frame': frame_index,
                'box': det['box'],
                'score': float(det['score'])
            })

    # Build pseudo ground truth per class via high-confidence detections + NMS
    # This approximates GT for mAP computation in absence of true annotations
    frame_gt_by_class = {}  # temporary per-frame GT
    # Group by class
    dets_by_class = {}
    for det in detections:
        if det['score'] >= PSEUDO_GT_THRESHOLD:
            cls = det['class_id']
            dets_by_class.setdefault(cls, {'boxes': [], 'scores': []})
            dets_by_class[cls]['boxes'].append(det['box'])
            dets_by_class[cls]['scores'].append(det['score'])
    # Apply NMS and store
    for cls, data in dets_by_class.items():
        boxes_c = data['boxes']
        scores_c = data['scores']
        keep_idx = nms_boxes(boxes_c, scores_c, iou_thresh=NMS_IOU_THRESHOLD)
        gt_boxes_kept = [boxes_c[k] for k in keep_idx]
        if len(gt_boxes_kept) > 0:
            # Update global GT dict
            gt_by_class.setdefault(cls, {})
            gt_by_class[cls].setdefault(frame_index, [])
            gt_by_class[cls][frame_index].extend(gt_boxes_kept)
            frame_gt_by_class[cls] = gt_boxes_kept

    # 4.4 Handle Output: Draw detections and overlay running mAP
    # Draw detections above CONF_THRESHOLD
    for det in detections:
        if det['score'] < CONF_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = det['box']
        cls = det['class_id']
        label = det['label']
        score = det['score']
        color = color_for_class(cls)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label_text = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, ymin - th - 4), (xmin + tw + 2, ymin), color, -1)
        cv2.putText(frame, label_text, (xmin + 1, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Compute and overlay running mAP
    running_map = compute_map(predictions_by_class, gt_by_class, iou_thresh=IOU_THRESHOLD)
    elapsed = time.time() - start_time
    fps_runtime = (frame_index / elapsed) if elapsed > 0 else 0.0
    info_text = f"mAP@IoU{IOU_THRESHOLD:.2f}: {running_map:.3f}  FPS: {fps_runtime:.1f}"
    cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # Write frame
    writer.write(frame)

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
writer.release()

# Final mAP over the whole video
final_map = compute_map(predictions_by_class, gt_by_class, iou_thresh=IOU_THRESHOLD)
print(f"Final pseudo mAP@IoU{IOU_THRESHOLD:.2f} with conf>={CONF_THRESHOLD}: {final_map:.4f}")
print(f"Output video saved to: {OUTPUT_PATH}")