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
confidence_threshold = 0.5

# IoU threshold used for evaluation (mAP) at 0.5 as common practice
IOU_THRESH_EVAL = 0.5
# Minimum score to consider a detection for evaluation accumulation (to reduce noise)
MIN_SCORE_FOR_AP = 0.05

# =========================
# Utility Functions
# =========================
def load_labels(path):
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Common label maps can be "index label" or just "label"
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1])
                else:
                    labels.append(line)
    except Exception as e:
        print("Failed to load labels:", e)
    return labels

def letterbox_resize(image, new_w, new_h):
    # For SSD models, typically a direct resize is used (no letterbox). Keep simple resize.
    return cv2.resize(image, (new_w, new_h))

def preprocess(frame_bgr, input_details):
    # Expect input format: [1, height, width, 3]
    _, in_h, in_w, _ = input_details[0]['shape']
    img_resized = letterbox_resize(frame_bgr, in_w, in_h)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_dtype = input_details[0]['dtype']

    if input_dtype == np.float32:
        input_data = (img_rgb.astype(np.float32) / 255.0).reshape(1, in_h, in_w, 3)
    else:
        # Assume uint8 quantized
        input_data = img_rgb.astype(np.uint8).reshape(1, in_h, in_w, 3)
    return input_data

def run_inference(interpreter, input_data, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Typical TFLite SSD outputs: boxes, classes, scores, num_detections
    # Retrieve by order based on output count and shapes
    outs = [interpreter.get_tensor(od['index']) for od in output_details]

    # Try to identify outputs by their shapes
    boxes = None
    classes = None
    scores = None
    num = None
    for out in outs:
        s = out.shape
        if len(s) == 3 and s[-1] == 4:
            boxes = out[0]
        elif len(s) == 2 and s[-1] >= 1 and s[0] == 1:
            # Could be classes or scores
            if out.dtype.kind in ('f',):  # float -> likely scores
                scores = out[0]
            else:
                classes = out[0]
        elif len(s) == 1 and s[0] == 1:
            num = int(np.squeeze(out))
    # Fallback if outputs are ordered as typical [boxes, classes, scores, num]
    if boxes is None or classes is None or scores is None or num is None:
        if len(outs) >= 4:
            boxes = np.squeeze(outs[0])
            classes = np.squeeze(outs[1])
            scores = np.squeeze(outs[2])
            num = int(np.squeeze(outs[3]))

    # Trim to num detections
    num = min(num, boxes.shape[0], classes.shape[0], scores.shape[0])
    return boxes[:num], classes[:num].astype(int), scores[:num].astype(float)

def box_iou(b1, b2):
    # Boxes in [ymin, xmin, ymax, xmax], normalized or absolute consistently
    y1 = max(b1[0], b2[0])
    x1 = max(b1[1], b2[1])
    y2 = min(b1[2], b2[2])
    x2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0

def cluster_gt_boxes(boxes, scores, iou_thresh=0.5):
    # Create GT-like boxes by clustering detections with IoU >= threshold.
    # Use highest-score box as the representative of each cluster.
    if len(boxes) == 0:
        return []
    idxs = np.argsort(-scores)  # descending
    clusters = []
    for idx in idxs:
        b = boxes[idx]
        matched = False
        for c in clusters:
            if box_iou(b, c) >= iou_thresh:
                matched = True
                break
        if not matched:
            clusters.append(b)
    return clusters

def evaluate_frame(d_boxes, d_classes, d_scores, per_class_data, iou_thresh=0.5, min_score=0.05):
    # per_class_data: dict[class_id] -> {'scores': [], 'tp': [], 'fp': [], 'gt': int}
    # Build pseudo-GT via clustering per class, then greedy matching.
    # Only consider detections with score >= min_score for evaluation.
    classes_in_frame = np.unique(d_classes.astype(int)).tolist()
    for c in classes_in_frame:
        # Filter detections of this class
        mask = (d_classes == c)
        boxes_c = d_boxes[mask]
        scores_c = d_scores[mask]

        # If no valid detections for class, skip
        if boxes_c.size == 0:
            continue

        # Build GT clusters
        gt_boxes = cluster_gt_boxes(boxes_c, scores_c, iou_thresh=iou_thresh)
        if c not in per_class_data:
            per_class_data[c] = {'scores': [], 'tp': [], 'fp': [], 'gt': 0}
        per_class_data[c]['gt'] += len(gt_boxes)

        # Prepare greedy matching
        matched = [False] * len(gt_boxes)
        # Consider detections for evaluation (above a low floor, not final display threshold)
        valid_idx = np.where(scores_c >= min_score)[0]
        if valid_idx.size == 0:
            continue
        # Sort by score desc
        order = valid_idx[np.argsort(-scores_c[valid_idx])]

        for i in order:
            det_box = boxes_c[i]
            det_score = scores_c[i]
            # Greedy match to gt
            best_iou = 0.0
            best_j = -1
            for j, gtb in enumerate(gt_boxes):
                if matched[j]:
                    continue
                iou = box_iou(det_box, gtb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                matched[best_j] = True
                per_class_data[c]['scores'].append(float(det_score))
                per_class_data[c]['tp'].append(1)
                per_class_data[c]['fp'].append(0)
            else:
                per_class_data[c]['scores'].append(float(det_score))
                per_class_data[c]['tp'].append(0)
                per_class_data[c]['fp'].append(1)

def compute_ap(rec, prec):
    # VOC 2010/2012 style AP: precision envelope and integrate over recall changes
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        if mpre[i-1] < mpre[i]:
            mpre[i-1] = mpre[i]
    # Identify points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)

def compute_map(per_class_data):
    aps = []
    for c, data in per_class_data.items():
        gt = int(data['gt'])
        if gt <= 0:
            continue
        if len(data['scores']) == 0:
            continue
        scores = np.array(data['scores'])
        tp = np.array(data['tp']).astype(np.float32)
        fp = np.array(data['fp']).astype(np.float32)
        # Sort by score descending
        order = np.argsort(-scores)
        tp = tp[order]
        fp = fp[order]
        # Cumulative
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        rec = cum_tp / float(gt)
        ap = compute_ap(rec, prec)
        aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

def make_color_for_class(c):
    # Deterministic simple palette
    r = (37 * (c + 1)) % 255
    g = (17 * (c + 1)) % 255
    b = (29 * (c + 1)) % 255
    # Avoid too-dark colors
    if r < 50 and g < 50 and b < 50:
        r = (r + 100) % 255
        g = (g + 100) % 255
        b = (b + 100) % 255
    return int(b), int(g), int(r)

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def get_video_info(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2 or np.isnan(fps):
        fps = 30.0
    return width, height, float(fps)

def draw_detections_on_frame(frame, boxes, classes, scores, labels, conf_thres, map_value):
    h, w = frame.shape[:2]
    for box, cls_id, score in zip(boxes, classes, scores):
        if score < conf_thres:
            continue
        ymin, xmin, ymax, xmax = box
        x1 = int(max(0, xmin) * w)
        y1 = int(max(0, ymin) * h)
        x2 = int(min(1, xmax) * w)
        y2 = int(min(1, ymax) * h)
        color = make_color_for_class(int(cls_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = str(cls_id)
        if 0 <= int(cls_id) < len(labels):
            label = labels[int(cls_id)]
        text = "{} {:.2f}".format(label, float(score))
        cv2.putText(frame, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # Overlay mAP
    map_text = "mAP@0.5 (proxy): {:.3f}".format(map_value)
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 220, 15), 2, cv2.LINE_AA)
    return frame

# =========================
# Main Processing
# =========================
def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # First pass: run inference, collect detections, accumulate for mAP
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Failed to open input video:", input_path)
        return
    vid_w, vid_h, vid_fps = get_video_info(cap)

    per_class_data = {}  # accumulation for mAP
    all_frame_detections = []  # store detections for drawing later

    frame_count = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_data = preprocess(frame, input_details)
        boxes, classes, scores = run_inference(interpreter, input_data, input_details, output_details)
        # boxes are normalized (ymin, xmin, ymax, xmax)
        # Store detections
        all_frame_detections.append({
            'boxes': boxes.copy(),
            'classes': classes.copy(),
            'scores': scores.copy()
        })
        # Accumulate for mAP (proxy using clustering as pseudo-GT)
        evaluate_frame(boxes, classes, scores, per_class_data, iou_thresh=IOU_THRESH_EVAL, min_score=MIN_SCORE_FOR_AP)

        frame_count += 1
    cap.release()
    t1 = time.time()
    elapsed = t1 - t0
    if elapsed <= 0:
        elapsed = 1e-6
    print("First pass: processed {} frames in {:.2f}s ({:.2f} FPS)".format(frame_count, elapsed, frame_count / elapsed))

    # Compute mAP (proxy)
    mAP = compute_map(per_class_data)
    print("Calculated mAP@0.5 (proxy, no ground-truth): {:.4f}".format(mAP))

    # Second pass: draw and save video
    ensure_dir_for_file(output_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Failed to reopen input video for writing:", input_path)
        return
    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (vid_w, vid_h))
    if not writer.isOpened():
        print("Failed to open output video for writing:", output_path)
        cap.release()
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(all_frame_detections):
            break
        det = all_frame_detections[frame_idx]
        frame_annotated = draw_detections_on_frame(
            frame,
            det['boxes'],
            det['classes'],
            det['scores'],
            labels,
            confidence_threshold,
            mAP
        )
        writer.write(frame_annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print("Output saved to:", output_path)

if __name__ == "__main__":
    main()