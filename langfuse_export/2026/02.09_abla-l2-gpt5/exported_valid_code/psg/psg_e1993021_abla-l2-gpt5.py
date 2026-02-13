import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5  # for drawing/filtering final detections
IOU_THRESHOLD = 0.5         # for matching/NMS
MIN_SCORE_FOR_MAP = 0.05    # keep low to accumulate predictions for proxy mAP

def load_labels(label_path):
    labels = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Common TFLite label files have '???' as label 0; skip if present
    if lines and lines[0].startswith('???'):
        lines = lines[1:]
    for i, name in enumerate(lines):
        labels[i] = name
    return labels

def preprocess_frame(frame, input_details):
    ih, iw = input_details['shape'][1], input_details['shape'][2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih))
    tensor = None
    if input_details['dtype'] == np.float32:
        tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    else:
        # uint8 (quantized) - pass as-is
        tensor = np.expand_dims(resized.astype(np.uint8), axis=0)
    return tensor

def set_input_tensor(interpreter, input_details, input_data):
    interpreter.set_tensor(input_details['index'], input_data)

def get_output_tensors(interpreter, output_details):
    outputs = []
    for od in output_details:
        outputs.append(interpreter.get_tensor(od['index']))
    # Expecting 4 outputs: boxes, classes, scores, count (typical SSD MobileNet V1)
    # Try to infer by shapes/dtypes if ordering is unknown
    boxes = classes = scores = num = None
    for out in outputs:
        out_shape = out.shape
        out_dtype = out.dtype
        if len(out_shape) == 2 and out_shape[1] == 4:
            boxes = out
        elif len(out_shape) == 2 and out_shape[1] > 1 and out_dtype in (np.float32, np.uint8):
            # could be classes or scores depending on dtype
            # classes is usually float32 but integral-like values
            # scores are float32 in [0,1]
            # Heuristic: if values are mostly integers within a small range, it's classes
            arr = out[0]
            if out_dtype == np.float32 and np.all(np.mod(arr, 1) == 0):
                classes = out
            else:
                # Try to distinguish by value range
                if np.max(arr) <= 1.0:
                    scores = out
                else:
                    classes = out
        elif len(out_shape) == 1 or (len(out_shape) == 2 and out_shape[1] == 1):
            num = out
    # Fallback to common TFLite ordering: [boxes, classes, scores, num]
    if boxes is None or classes is None or scores is None:
        try:
            boxes, classes, scores, num = outputs
        except Exception:
            pass
    return boxes, classes, scores, num

def denorm_box_to_pixels(box, frame_w, frame_h):
    # TFLite SSD outputs boxes as [ymin, xmin, ymax, xmax] normalized [0,1]
    ymin, xmin, ymax, xmax = box
    xmin = int(max(0, min(frame_w - 1, xmin * frame_w)))
    xmax = int(max(0, min(frame_w - 1, xmax * frame_w)))
    ymin = int(max(0, min(frame_h - 1, ymin * frame_h)))
    ymax = int(max(0, min(frame_h - 1, ymax * frame_h)))
    return [xmin, ymin, xmax, ymax]

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    idxs = list(np.argsort(scores)[::-1])
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if iou(boxes[current], boxes[i]) < iou_thresh]
    return keep

def build_proxy_ground_truth(all_detections, num_classes, confidence_threshold=0.5, iou_thresh=0.5):
    # Returns gt_by_class: {class_id: {frame_idx: [boxes...]}}
    gt_by_class = {c: {} for c in range(num_classes)}
    for frame_idx, frame_dets in enumerate(all_detections):
        # Group by class
        by_class = {}
        for det in frame_dets:
            c = det['class_id']
            s = det['score']
            if s < confidence_threshold:
                continue
            by_class.setdefault(c, {'boxes': [], 'scores': []})
            by_class[c]['boxes'].append(det['box'])
            by_class[c]['scores'].append(s)
        # NMS per class to get proxy GT boxes
        for c, data in by_class.items():
            boxes = data['boxes']
            scores = data['scores']
            keep_idx = nms(boxes, scores, iou_thresh=iou_thresh)
            kept_boxes = [boxes[i] for i in keep_idx]
            if kept_boxes:
                gt_by_class[c].setdefault(frame_idx, [])
                gt_by_class[c][frame_idx].extend(kept_boxes)
    return gt_by_class

def compute_map_proxy(all_detections, num_classes, iou_thresh=0.5, gt_conf_thresh=0.5, min_pred_score=0.05):
    # Build proxy ground-truth using NMS on confident detections
    gt_by_class = build_proxy_ground_truth(all_detections, num_classes, confidence_threshold=gt_conf_thresh, iou_thresh=iou_thresh)

    ap_list = []

    for c in range(num_classes):
        # Build predictions list across frames for this class
        preds = []
        for frame_idx, frame_dets in enumerate(all_detections):
            for det in frame_dets:
                if det['class_id'] == c and det['score'] >= min_pred_score:
                    preds.append({'frame_idx': frame_idx, 'score': det['score'], 'box': det['box']})
        if len(preds) == 0:
            continue

        # Total number of GT boxes for this class
        total_gt = sum(len(lst) for lst in gt_by_class.get(c, {}).values())
        if total_gt == 0:
            # No GT proxies for this class; skip from mAP
            continue

        # Sort predictions by descending score
        preds.sort(key=lambda x: x['score'], reverse=True)

        # For each frame, create 'used' flags for GT boxes
        used_flags = {}
        for fidx, gts in gt_by_class.get(c, {}).items():
            used_flags[fidx] = [False] * len(gts)

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, p in enumerate(preds):
            fidx = p['frame_idx']
            pred_box = p['box']
            gts = gt_by_class.get(c, {}).get(fidx, [])
            best_iou = 0.0
            best_idx = -1
            for j, gt_box in enumerate(gts):
                if used_flags[fidx][j]:
                    continue
                iou_val = iou(pred_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j
            if best_iou >= iou_thresh and best_idx >= 0:
                tp[i] = 1.0
                used_flags[fidx][best_idx] = True
            else:
                fp[i] = 1.0

        # Cumulative precision-recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / max(1, total_gt)
        precision = cum_tp / np.maximum(1, (cum_tp + cum_fp))

        # Compute AP using precision envelope method
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        # Sum over recall steps where it changes
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        ap_list.append(ap)

    if len(ap_list) == 0:
        return None
    return float(np.mean(ap_list))

def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)
    num_classes = max(labels.keys()) + 1 if labels else 0

    # Load TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    # Open input video (single file)
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Error: Cannot open input video:", INPUT_PATH)
        return

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # First pass: run inference and store detections per frame
    all_detections = []  # list of lists; one per frame
    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame, input_details)
        set_input_tensor(interpreter, input_details, input_tensor)
        interpreter.invoke()
        boxes_out, classes_out, scores_out, num_out = get_output_tensors(interpreter, output_details)

        # Normalize shapes
        if boxes_out is None or classes_out is None or scores_out is None:
            print("Error: Unexpected TFLite model outputs.")
            cap.release()
            return

        # Squeeze to remove batch dimension
        boxes = np.squeeze(boxes_out)
        classes = np.squeeze(classes_out).astype(np.int32)
        scores = np.squeeze(scores_out).astype(np.float32)

        # Some models provide num_detections
        if num_out is not None:
            n = int(np.squeeze(num_out))
            boxes = boxes[:n]
            classes = classes[:n]
            scores = scores[:n]

        frame_dets = []
        for i in range(len(scores)):
            score = float(scores[i])
            cls_id = int(classes[i])
            # Clip class id to labels range if necessary
            if labels and cls_id not in labels:
                # If class id exceeds label mapping, skip
                continue
            # Convert normalized box to pixel coords
            box_norm = boxes[i]
            xmin, ymin, xmax, ymax = denorm_box_to_pixels(box_norm, width, height)
            # Sanity check
            if xmax <= xmin or ymax <= ymin:
                continue
            frame_dets.append({
                'class_id': cls_id,
                'score': score,
                'box': [xmin, ymin, xmax, ymax]
            })

        all_detections.append(frame_dets)
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    if frame_idx > 0:
        print(f"Inference pass complete: {frame_idx} frames in {elapsed:.2f}s ({frame_idx / max(1e-6, elapsed):.2f} FPS)")

    # Compute proxy mAP across the video
    map_score = compute_map_proxy(
        all_detections=all_detections,
        num_classes=num_classes,
        iou_thresh=IOU_THRESHOLD,
        gt_conf_thresh=CONFIDENCE_THRESHOLD,
        min_pred_score=MIN_SCORE_FOR_MAP
    )

    # Second pass: draw detections and save output video with mAP overlay
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Error: Cannot reopen input video for writing:", INPUT_PATH)
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Error: Cannot open output video for writing:", OUTPUT_PATH)
        cap.release()
        return

    frame_index = 0
    overlay_text = f"mAP: {'N/A' if map_score is None else f'{map_score:.3f}'}"
    text_pos = (10, 25)
    label_font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret or frame_index >= len(all_detections):
            break

        dets = all_detections[frame_index]
        # Filter for drawing using confidence threshold
        draw_dets = [d for d in dets if d['score'] >= CONFIDENCE_THRESHOLD]

        for det in draw_dets:
            xmin, ymin, xmax, ymax = det['box']
            cls_id = det['class_id']
            score = det['score']
            label = labels.get(cls_id, str(cls_id)) if labels else str(cls_id)

            # Draw rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Prepare label text
            text = f"{label}: {score:.2f}"
            (tw, th), bl = cv2.getTextSize(text, label_font, 0.5, 1)
            # Draw filled rectangle behind text for readability
            cv2.rectangle(frame, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, ymin), (0, 255, 0), -1)
            cv2.putText(frame, text, (xmin + 2, max(0, ymin - 4)), label_font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw mAP overlay
        cv2.putText(frame, overlay_text, text_pos, label_font, 0.7, (50, 200, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print("Output saved to:", OUTPUT_PATH)
    if map_score is None:
        print("mAP: N/A (no sufficient proxy ground-truth could be derived)")
    else:
        print(f"mAP (proxy, GT via high-conf NMS): {map_score:.3f}")

if __name__ == "__main__":
    main()