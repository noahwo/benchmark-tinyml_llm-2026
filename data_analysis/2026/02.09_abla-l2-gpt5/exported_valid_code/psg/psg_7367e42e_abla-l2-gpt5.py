import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters (provided)
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5  # for drawing/final results

# Helper: ensure output directory exists
def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

# Load labels from file
def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

# Prepare input tensor from frame according to interpreter input details
def prepare_input(frame_bgr, input_details):
    # Determine input shape and dtype
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
    dtype = input_details[0]['dtype']
    quant = input_details[0].get('quantization', (0.0, 0))
    scale, zero_point = quant if isinstance(quant, (tuple, list)) else (0.0, 0)

    # Convert BGR to RGB and resize
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    if dtype == np.uint8:
        if scale and scale > 0:
            # Quantize if quantization params provided
            input_data = np.clip(resized.astype(np.float32) / scale + zero_point, 0, 255).astype(np.uint8)
        else:
            input_data = resized.astype(np.uint8)
    else:
        # float32 path
        input_data = resized.astype(np.float32) / 255.0

    input_data = np.expand_dims(input_data, axis=0)
    return input_data

# Retrieve model outputs robustly
def get_detection_outputs(interpreter, output_details):
    # Attempt to map by names if available, else by shape heuristics
    tensors = {}
    for od in output_details:
        name = od.get('name', '').lower()
        data = interpreter.get_tensor(od['index'])
        if 'box' in name:
            tensors['boxes'] = data
        elif 'score' in name:
            tensors['scores'] = data
        elif 'class' in name:
            tensors['classes'] = data
        elif 'num' in name or 'count' in name:
            tensors['num'] = data

    # Fallback using shapes if names were not informative
    if 'boxes' not in tensors or 'scores' not in tensors or 'classes' not in tensors or 'num' not in tensors:
        outs = {od['index']: interpreter.get_tensor(od['index']) for od in output_details}
        # boxes: 3D with last dim=4
        for k, v in outs.items():
            if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 4:
                tensors['boxes'] = v
        # scores/classes: 2D [1, N]
        candidates_2d = [v for v in outs.values() if isinstance(v, np.ndarray) and v.ndim == 2]
        if len(candidates_2d) >= 2:
            # Heuristic: classes are closer to integer values
            a, b = candidates_2d[0], candidates_2d[1]
            def is_int_like(x):
                return np.mean(np.abs(x - np.round(x))) < 1e-3
            if is_int_like(a) and not is_int_like(b):
                tensors['classes'] = a
                tensors['scores'] = b
            elif is_int_like(b) and not is_int_like(a):
                tensors['classes'] = b
                tensors['scores'] = a
            else:
                # If both look similar, choose by name if exists, else default order
                tensors.setdefault('scores', a)
                tensors.setdefault('classes', b)
        # num: 1D [1]
        for v in outs.values():
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == 1:
                tensors['num'] = v

    # Final safety checks
    boxes = tensors.get('boxes', None)
    classes = tensors.get('classes', None)
    scores = tensors.get('scores', None)
    num = tensors.get('num', None)
    if boxes is None or classes is None or scores is None or num is None:
        raise RuntimeError("Could not parse TFLite detection outputs.")

    # Squeeze to remove batch dimension
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    num = int(np.squeeze(num).astype(np.int32))
    return boxes, classes, scores, num

# Convert normalized box [ymin, xmin, ymax, xmax] to absolute [xmin, ymin, xmax, ymax] in pixels and clamp
def denormalize_and_clamp(box, img_w, img_h):
    ymin, xmin, ymax, xmax = box
    xmin_abs = max(0, min(int(xmin * img_w), img_w - 1))
    xmax_abs = max(0, min(int(xmax * img_w), img_w - 1))
    ymin_abs = max(0, min(int(ymin * img_h), img_h - 1))
    ymax_abs = max(0, min(int(ymax * img_h), img_h - 1))
    # Ensure proper ordering
    if xmax_abs < xmin_abs: xmin_abs, xmax_abs = xmax_abs, xmin_abs
    if ymax_abs < ymin_abs: ymin_abs, ymax_abs = ymax_abs, ymin_abs
    return [xmin_abs, ymin_abs, xmax_abs, ymax_abs]

# Compute IoU between a single box and an array of boxes
def iou_with_many(box, boxes):
    # box, boxes expected as [xmin, ymin, xmax, ymax]
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])

    interW = np.maximum(0, xB - xA + 1)
    interH = np.maximum(0, yB - yA + 1)
    interArea = interW * interH

    boxArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    boxesArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    unionArea = boxArea + boxesArea - interArea + 1e-9
    return interArea / unionArea

# Greedy Non-Maximum Suppression
def nms(boxes, scores, iou_threshold=0.5):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    idxs = scores.argsort()[::-1]
    selected = []
    while idxs.size > 0:
        i = idxs[0]
        selected.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = iou_with_many(boxes[i], boxes[rest])
        keep = rest[ious <= iou_threshold]
        idxs = keep
    return np.array(selected, dtype=np.int32)

# Compute AP for a single class given GT boxes and predicted boxes with scores
def compute_ap(gt_boxes, pred_boxes, pred_scores, iou_thresh=0.5):
    # gt_boxes: (G, 4), pred_boxes: (P, 4), pred_scores: (P,)
    G = gt_boxes.shape[0]
    if G == 0:
        return None  # undefined; will be skipped in mAP
    if pred_boxes.shape[0] == 0:
        return 0.0

    # Sort predictions by score desc
    order = pred_scores.argsort()[::-1]
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    matched_gt = np.zeros((G,), dtype=np.uint8)
    tp = np.zeros((pred_boxes.shape[0],), dtype=np.float32)
    fp = np.zeros((pred_boxes.shape[0],), dtype=np.float32)

    for i in range(pred_boxes.shape[0]):
        ious = iou_with_many(pred_boxes[i], gt_boxes)
        best_idx = int(np.argmax(ious)) if ious.size > 0 else -1
        best_iou = ious[best_idx] if ious.size > 0 else 0.0
        if best_iou >= iou_thresh and matched_gt[best_idx] == 0:
            tp[i] = 1.0
            matched_gt[best_idx] = 1
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / (G + 1e-9)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # Precision envelope
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]
    # Integrate AP as area under curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

# Compute mAP across classes (skipping classes with no GT)
def compute_map(gt_by_class, pred_by_class, iou_thresh=0.5):
    aps = []
    for cls_id in gt_by_class.keys():
        gt_list = gt_by_class.get(cls_id, [])
        pred_list = pred_by_class.get(cls_id, [])
        if len(gt_list) == 0:
            continue
        gt_boxes = np.array(gt_list, dtype=np.float32).reshape(-1, 4)
        if len(pred_list) == 0:
            aps.append(0.0)
            continue
        pred_boxes = np.array([p[1] for p in pred_list], dtype=np.float32).reshape(-1, 4)
        pred_scores = np.array([p[0] for p in pred_list], dtype=np.float32).reshape(-1,)
        ap = compute_ap(gt_boxes, pred_boxes, pred_scores, iou_thresh=iou_thresh)
        if ap is not None:
            aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# Deterministic color for a class id
def class_color(cls_id):
    # Simple hashing to BGR
    r = (37 * (cls_id + 1)) % 255
    g = (17 * (cls_id + 1)) % 255
    b = (29 * (cls_id + 1)) % 255
    return int(b), int(g), int(r)

def main():
    # Setup
    ensure_parent_dir(OUTPUT_PATH)
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video: %s" + INPUT_PATH)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, float(fps), (in_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open output video for writing: %s" + OUTPUT_PATH)

    # Aggregators for proxy mAP (GT via NMS-filtered detections; Predictions = raw detections)
    gt_by_class = {}    # cls_id -> list of boxes
    pred_by_class = {}  # cls_id -> list of (score, box)

    # Inference loop
    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Prepare input
        input_data = prepare_input(frame, input_details)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Inference
        interpreter.invoke()

        # Get outputs
        boxes_norm, classes_raw, scores_raw, num = get_detection_outputs(interpreter, output_details)

        # Collect detections
        # Use a lower threshold for prediction collection (for PR curve)
        raw_thresh = 0.05
        preds_per_class_boxes = {}
        preds_per_class_scores = {}

        drawn_boxes = []  # final boxes to draw after NMS at CONFIDENCE_THRESHOLD
        drawn_scores = []
        drawn_classes = []

        # Make sure we iterate only over 'num'
        det_count = min(num, boxes_norm.shape[0], scores_raw.shape[0], classes_raw.shape[0])
        for i in range(det_count):
            score = float(scores_raw[i])
            cls_id = int(classes_raw[i])
            # Map to label index: many TFLite SSD labelmaps have '???' at 0, so add 1
            mapped_cls_id = cls_id  # store original numeric id for metrics
            # Denormalize box
            abs_box = denormalize_and_clamp(boxes_norm[i], in_w, in_h)  # [xmin, ymin, xmax, ymax]

            # Collect raw predictions per class for mAP computation
            if score >= raw_thresh:
                if mapped_cls_id not in preds_per_class_boxes:
                    preds_per_class_boxes[mapped_cls_id] = []
                    preds_per_class_scores[mapped_cls_id] = []
                preds_per_class_boxes[mapped_cls_id].append(abs_box)
                preds_per_class_scores[mapped_cls_id].append(score)

            # Collect for final drawing if above main threshold
            if score >= CONFIDENCE_THRESHOLD:
                drawn_boxes.append(abs_box)
                drawn_scores.append(score)
                drawn_classes.append(mapped_cls_id)

        # Update prediction aggregator
        for c in preds_per_class_boxes.keys():
            if c not in pred_by_class:
                pred_by_class[c] = []
            for pb, ps in zip(preds_per_class_boxes[c], preds_per_class_scores[c]):
                pred_by_class[c].append((ps, pb))

        # Build GT via NMS on drawn detections per class
        gt_this_frame_by_class = {}
        if len(drawn_boxes) > 0:
            drawn_boxes_arr = np.array(drawn_boxes, dtype=np.float32)
            drawn_scores_arr = np.array(drawn_scores, dtype=np.float32)
            drawn_classes_arr = np.array(drawn_classes, dtype=np.int32)
            # Per class NMS
            for c in np.unique(drawn_classes_arr):
                mask = (drawn_classes_arr == c)
                c_boxes = drawn_boxes_arr[mask]
                c_scores = drawn_scores_arr[mask]
                keep_idx = nms(c_boxes, c_scores, iou_threshold=0.5)
                if keep_idx.size > 0:
                    c_kept = c_boxes[keep_idx]
                    gt_this_frame_by_class[c] = c_kept
        # Update GT aggregator
        for c, c_gt_boxes in gt_this_frame_by_class.items():
            if c not in gt_by_class:
                gt_by_class[c] = []
            for b in c_gt_boxes:
                gt_by_class[c].append(b.tolist())

        # Compute running mAP (proxy) across all frames so far
        map_val = compute_map(gt_by_class, pred_by_class, iou_thresh=0.5)

        # Draw final (NMS-filtered) boxes and labels on frame
        # We already have NMS per class results in gt_this_frame_by_class; draw those
        for c, c_gt_boxes in gt_this_frame_by_class.items():
            color = class_color(c)
            for b in c_gt_boxes:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Choose the highest score among raw predictions overlapping this GT for label score display
                # Find best-matching prediction for score display
                scores_for_c = preds_per_class_scores.get(c, [])
                boxes_for_c = preds_per_class_boxes.get(c, [])
                best_score = 0.0
                if boxes_for_c:
                    boxes_for_c_arr = np.array(boxes_for_c, dtype=np.float32)
                    ious = iou_with_many([x1, y1, x2, y2], boxes_for_c_arr)
                    if ious.size > 0:
                        j = int(np.argmax(ious))
                        best_score = float(scores_for_c[j])
                # Label text
                # Use label mapping with +1 offset if appropriate
                lbl_idx = c + 1 if (c + 1) < len(labels) else c
                lbl_text = labels[lbl_idx] if lbl_idx < len(labels) else f"id:{c}"
                text = f"{lbl_text} {best_score*100:.1f}%"
                cv2.putText(frame, text, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Overlay mAP
        cv2.putText(frame, f"mAP: {map_val*100:.2f}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)

        # Write frame
        writer.write(frame)

    # Cleanup
    cap.release()
    writer.release()

    total_time = time.time() - t0
    final_map = compute_map(gt_by_class, pred_by_class, iou_thresh=0.5)
    print("Processing complete.")
    print(f"Frames processed: {frame_idx}")
    print(f"Elapsed time: {total_time:.2f}s")
    print(f"Proxy mAP (detections vs. NMS-filtered detections) @ IoU=0.5: {final_map:.4f}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()