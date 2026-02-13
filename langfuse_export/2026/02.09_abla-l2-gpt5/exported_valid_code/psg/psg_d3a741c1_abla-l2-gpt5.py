import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5  # only keep detections with score >= threshold


# =========================
# Utility Functions
# =========================
def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                labels[i] = line
    # Determine label offset if label file has '???' at index 0 (common for TF SSD label maps)
    label_offset = 1 if (0 in labels and labels[0] == "???") else 0
    return labels, label_offset


def get_quant_params(tensor_detail):
    # Try standard TFLite 'quantization' tuple
    q = tensor_detail.get('quantization', None)
    if q is not None and isinstance(q, tuple) and len(q) == 2:
        scale, zero_point = q
        if scale is not None and zero_point is not None:
            return float(scale), int(zero_point)
    # Try 'quantization_parameters' dict
    qp = tensor_detail.get('quantization_parameters', None)
    if qp:
        scales = qp.get('scales', None)
        zeros = qp.get('zero_points', None)
        if scales is not None and len(scales) > 0 and zeros is not None and len(zeros) > 0:
            return float(scales[0]), int(zeros[0])
    return None, None


def preprocess_frame(frame_bgr, input_shape, input_dtype, input_scale=None, input_zero_point=None):
    # input_shape: [1, height, width, 3]
    h, w = int(input_shape[1]), int(input_shape[2])
    resized = cv2.resize(frame_bgr, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        inp = (rgb.astype(np.float32) / 255.0)
    elif input_dtype == np.uint8:
        if input_scale is not None and input_zero_point is not None and input_scale > 0:
            inp = rgb.astype(np.float32) / input_scale + input_zero_point
            inp = np.clip(np.rint(inp), 0, 255).astype(np.uint8)
        else:
            inp = rgb.astype(np.uint8)
    else:
        # Fallback to float32 normalization
        inp = (rgb.astype(np.float32) / 255.0)
    inp = np.expand_dims(inp, axis=0)
    return inp


def iou_xyxy(boxA, boxB):
    # boxes: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_per_class(boxes, scores, class_ids, iou_thresh=0.5):
    # Perform per-class NMS; return indices to keep
    indices = np.arange(len(boxes))
    if len(indices) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids, dtype=np.int32)

    keep_indices = []
    for c in np.unique(class_ids):
        c_mask = (class_ids == c)
        c_inds = indices[c_mask]
        if len(c_inds) == 0:
            continue
        c_boxes = boxes[c_mask]
        c_scores = scores[c_mask]
        order = np.argsort(-c_scores)
        c_inds = c_inds[order]
        c_boxes = c_boxes[order]

        kept = []
        while len(c_inds) > 0:
            i = c_inds[0]
            kept.append(i)
            if len(c_inds) == 1:
                break
            rest = c_inds[1:]
            rest_boxes = c_boxes[1:]
            i_box = c_boxes[0]
            ious = np.array([iou_xyxy(i_box, rb) for rb in rest_boxes], dtype=np.float32)
            remain_mask = ious <= iou_thresh
            c_inds = rest[remain_mask]
            c_boxes = rest_boxes[remain_mask]
        keep_indices.extend(kept)
    return keep_indices


def color_for_class(class_id):
    # Deterministic color for a class id (BGR)
    rng = (class_id * 123457) % 0xFFFFFF
    b = 50 + (rng & 0xFF) % 206
    g = 50 + ((rng >> 8) & 0xFF) % 206
    r = 50 + ((rng >> 16) & 0xFF) % 206
    return (int(b), int(g), int(r))


def compute_ap_11pt(precisions, recalls):
    # VOC 2007 11-point AP
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p = 0.0
        for pr, rc in zip(precisions, recalls):
            if rc >= t and pr > p:
                p = pr
        ap += p
    return ap / 11.0


def compute_map(predictions_per_class, gt_totals_per_class):
    # predictions_per_class: dict[class_id] -> list of (score, is_tp)
    # gt_totals_per_class: dict[class_id] -> int total GT count
    ap_list = []
    for c, preds in predictions_per_class.items():
        gt_total = gt_totals_per_class.get(c, 0)
        if gt_total <= 0 or len(preds) == 0:
            continue
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        for score, is_tp in preds_sorted:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / max(tp_cum + fp_cum, 1))
            recalls.append(tp_cum / gt_total if gt_total > 0 else 0.0)
        ap = compute_ap_11pt(precisions, recalls)
        ap_list.append(ap)
    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(ap_list))


# =========================
# Initialize TFLite Interpreter
# =========================
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
in_scale, in_zero = get_quant_params(input_details[0])

# =========================
# Load labels
# =========================
labels, label_offset = load_labels(label_path)

# =========================
# Video IO
# =========================
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Failed to open input video: {}".format(input_path))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ensure_dir_for_file(output_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not writer.isOpened():
    raise RuntimeError("Failed to open output video for writing: {}".format(output_path))

# =========================
# mAP (Proxy) Accumulators
# =========================
# We estimate AP/mAP using temporal consistency: treat previous frame's detections as pseudo-GT for current frame.
predictions_per_class = {}  # class_id -> list of (score, is_tp)
gt_totals_per_class = {}    # class_id -> total pseudo-GT boxes count
prev_dets_by_class = {}     # class_id -> list of boxes [x1,y1,x2,y2] from previous frame

frame_index = 0
t_start = time.time()

# =========================
# Process video frames
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    inp = preprocess_frame(frame, input_shape, input_dtype, in_scale, in_zero)

    # Inference
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()

    # Postprocess: extract outputs (assuming standard SSD order)
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num_det = interpreter.get_tensor(output_details[3]['index'])
    except Exception:
        # Fallback: retrieve all and guess
        outs = [interpreter.get_tensor(od['index']) for od in output_details]
        # Identify boxes (has last dim 4)
        boxes = None
        classes = None
        scores = None
        num_det = None
        for arr in outs:
            shp = arr.shape
            if len(shp) >= 2 and shp[-1] == 4:
                boxes = arr
        for arr in outs:
            shp = arr.shape
            if len(shp) == 2 and boxes is not None and shp[1] == boxes.shape[1]:
                # Could be classes or scores; determine by dtype/values
                if arr.dtype in (np.float32, np.float64) and np.all((arr >= 0) & (arr <= 1)):
                    scores = arr
                else:
                    classes = arr
            if len(shp) == 1 and shp[0] == 1:
                num_det = arr
        if boxes is None or classes is None or scores is None or num_det is None:
            raise RuntimeError("Unable to parse model outputs.")

    # Remove batch dimension if present
    if len(boxes.shape) == 3:
        boxes = boxes[0]
    if len(classes.shape) >= 2:
        classes = classes[0]
    if len(scores.shape) >= 2:
        scores = scores[0]
    if hasattr(num_det, "__len__"):
        nd = int(num_det[0]) if len(num_det) > 0 else int(num_det)
    else:
        nd = int(num_det)

    # Convert detections to lists
    detections_xyxy = []
    detection_scores = []
    detection_class_ids = []

    for i in range(nd):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        cls_raw = int(classes[i])
        label_id = cls_raw + label_offset
        # clamp label id if out of range
        if label_id not in labels:
            # fallback to raw if offset overflows
            label_id = cls_raw if cls_raw in labels else cls_raw

        y_min, x_min, y_max, x_max = boxes[i]
        # Coordinates are normalized [0,1]
        x1 = int(max(0, min(1, x_min)) * width)
        y1 = int(max(0, min(1, y_min)) * height)
        x2 = int(max(0, min(1, x_max)) * width)
        y2 = int(max(0, min(1, y_max)) * height)
        # Ensure proper ordering
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        # Skip degenerate boxes
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue

        detections_xyxy.append([x1, y1, x2, y2])
        detection_scores.append(score)
        detection_class_ids.append(label_id)

    # Apply per-class NMS
    keep = nms_per_class(detections_xyxy, detection_scores, detection_class_ids, iou_thresh=0.5)
    detections_xyxy = [detections_xyxy[i] for i in keep]
    detection_scores = [detection_scores[i] for i in keep]
    detection_class_ids = [detection_class_ids[i] for i in keep]

    # Draw detections
    for box, sc, cid in zip(detections_xyxy, detection_scores, detection_class_ids):
        x1, y1, x2, y2 = box
        color = color_for_class(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_name = labels.get(cid, str(cid))
        text = "{}: {:.2f}".format(label_name, sc)
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Update proxy mAP stats using temporal matching (previous frame as pseudo-GT)
    curr_by_class = {}
    for box, sc, cid in zip(detections_xyxy, detection_scores, detection_class_ids):
        curr_by_class.setdefault(cid, []).append((box, sc))

    if frame_index > 0:
        # For each class, match current preds to previous frame boxes (pseudo-GT) via IoU > 0.5
        for cid in set(list(prev_dets_by_class.keys()) + list(curr_by_class.keys())):
            prev_boxes = prev_dets_by_class.get(cid, [])
            gt_totals_per_class[cid] = gt_totals_per_class.get(cid, 0) + len(prev_boxes)
            curr_list = curr_by_class.get(cid, [])
            matched_prev = [False] * len(prev_boxes)

            # Sort current detections by score descending to simulate standard matching
            curr_list_sorted = sorted(curr_list, key=lambda x: x[1], reverse=True)

            for curr_box, curr_score in curr_list_sorted:
                # find best match
                best_iou = 0.0
                best_j = -1
                for j, gt_box in enumerate(prev_boxes):
                    if matched_prev[j]:
                        continue
                    iou = iou_xyxy(curr_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                is_tp = False
                if best_j >= 0 and best_iou >= 0.5:
                    matched_prev[best_j] = True
                    is_tp = True
                predictions_per_class.setdefault(cid, []).append((curr_score, is_tp))
    # Set current as previous for next iteration
    prev_dets_by_class = {cid: [b for (b, s) in curr_by_class.get(cid, [])] for cid in curr_by_class.keys()}

    # Compute running mAP (proxy) and overlay
    running_map = compute_map(predictions_per_class, gt_totals_per_class)
    map_text = "mAP (proxy): {:.3f}".format(running_map)
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Write frame
    writer.write(frame)
    frame_index += 1

# Cleanup
cap.release()
writer.release()

# Final mAP (proxy)
final_map = compute_map(predictions_per_class, gt_totals_per_class)

# Print summary
elapsed = time.time() - t_start
print("Processed {} frames in {:.2f}s ({:.2f} FPS).".format(frame_index, elapsed, frame_index / max(elapsed, 1e-6)))
print("Output saved to:", output_path)
print("Final mAP (proxy): {:.4f}".format(final_map))