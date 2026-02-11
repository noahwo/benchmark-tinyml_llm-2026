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
confidence_threshold = 0.5  # for drawing on output video

# Pseudo-GT and evaluation thresholds for mAP computation
PSEUDO_GT_CONF_THRESHOLD = 0.8   # detections >= this are treated as pseudo ground-truth
EVAL_MIN_CONF_THRESHOLD = 0.0    # include all predictions with score >= this for PR curve
IOU_THRESHOLD = 0.5              # IoU threshold for matching predictions to GT (mAP calc)


def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lbl = line.strip()
            if lbl:
                labels.append(lbl)
    return labels


def make_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_input_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    idx = input_details['index']
    shape = input_details['shape']
    dtype = input_details['dtype']
    return idx, shape, dtype


def preprocess_frame(frame_bgr, input_shape, input_dtype):
    # input_shape: [1, h, w, 3]
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    resized = cv2.resize(frame_bgr, (in_w, in_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.uint8)
    return np.expand_dims(rgb, axis=0)


def parse_tflite_outputs(interpreter):
    out_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(od['index']) for od in out_details]
    boxes = None
    classes = None
    scores = None
    num = None

    # Identify outputs by shape/value characteristics
    for arr in outputs:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr[0]
    # Remaining arrays
    rem = [arr for arr in outputs if arr is not None and arr is not boxes]
    # num_detections (size 1)
    for arr in rem:
        if arr.size == 1:
            num = int(round(float(arr.flatten()[0])))
            break
    # remove num from remaining
    rem2 = [arr for arr in rem if arr is not num]

    # Two arrays left: classes and scores, both typically shape (1, N)
    # Scores are in [0,1], classes are >= 0 and often > 1
    for arr in rem2:
        a = arr[0] if arr.ndim == 2 else arr
        m = float(np.max(a)) if a.size else -1.0
        if m <= 1.0:
            scores = a
        else:
            classes = a

    # Fallbacks in case ordering differs
    if boxes is None:
        # Try to find by last dim 4 even if not ndims=3
        for arr in outputs:
            if arr.shape[-1] == 4:
                boxes = arr.reshape((-1, 4))
                break
    if classes is None or scores is None:
        for arr in outputs:
            if arr is boxes or arr is num:
                continue
            a = arr[0] if arr.ndim == 2 else arr
            if a.size == 0:
                continue
            m = float(np.max(a))
            if classes is None and m > 1.0:
                classes = a
            elif scores is None and m <= 1.0:
                scores = a

    # Final safety conversions
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.float32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)
    if num is None:
        num = min(len(scores), len(classes), len(boxes))

    # Ensure lengths align
    num = min(num, len(scores), len(classes), len(boxes))
    return boxes[:num], classes[:num], scores[:num], num


def convert_boxes_to_pixels(boxes_norm, frame_w, frame_h):
    # boxes_norm: [N, 4] in [ymin, xmin, ymax, xmax], normalized [0,1]
    boxes_px = []
    for y_min, x_min, y_max, x_max in boxes_norm:
        x1 = int(max(0, min(frame_w - 1, x_min * frame_w)))
        y1 = int(max(0, min(frame_h - 1, y_min * frame_h)))
        x2 = int(max(0, min(frame_w - 1, x_max * frame_w)))
        y2 = int(max(0, min(frame_h - 1, y_max * frame_h)))
        boxes_px.append((x1, y1, x2, y2))
    return boxes_px


def iou(boxA, boxB):
    # boxes as (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = (boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def compute_map(pseudo_gt_by_cls, preds_by_cls, iou_threshold=IOU_THRESHOLD):
    # Compute AP per class and return their mean (over classes with at least one GT)
    ap_list = []

    for cls_id, gt_list in pseudo_gt_by_cls.items():
        gt_count = len(gt_list)
        if gt_count == 0:
            continue

        preds = preds_by_cls.get(cls_id, [])
        # preds: list of (score, frame_id, box)
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)

        # For each frame, keep a list of gt boxes and which ones matched
        gt_by_frame = {}
        for frame_id, gt_box in gt_list:
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            gt_by_frame[frame_id].append({'box': gt_box, 'matched': False})

        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        for i, (score, frame_id, pred_box) in enumerate(preds_sorted):
            gts = gt_by_frame.get(frame_id, [])
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if g['matched']:
                    continue
                iou_val = iou(pred_box, g['box'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_iou >= iou_threshold and best_j >= 0:
                gts[best_j]['matched'] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        # Precision-Recall curve
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(gt_count, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        # AP computation (VOC-style 11-point interpolation)
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            prec_at_recall = precisions[recalls >= t]
            p = np.max(prec_at_recall) if prec_at_recall.size > 0 else 0.0
            ap += p / 11.0

        ap_list.append(ap)

    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(ap_list))


def ensure_dir_for_file(file_path):
    d = os.path.dirname(file_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    labels = load_labels(label_path)
    interpreter = make_interpreter(model_path)
    input_index, input_shape, input_dtype = get_input_details(interpreter)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # First pass: run inference, collect predictions and pseudo-GT for mAP computation
    pseudo_gt_by_cls = {}  # {cls_id: [(frame_id, box), ...]}
    preds_by_cls = {}      # {cls_id: [(score, frame_id, box), ...]}

    frame_idx = 0
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame, input_shape, input_dtype)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        boxes_norm, classes_raw, scores, num = parse_tflite_outputs(interpreter)
        boxes_px = convert_boxes_to_pixels(boxes_norm, width, height)

        # Fill pseudo-GT and prediction pools
        for i in range(num):
            cls_id = int(classes_raw[i])
            score = float(scores[i])
            box = boxes_px[i]

            # Predictions for evaluation
            if score >= EVAL_MIN_CONF_THRESHOLD:
                preds_by_cls.setdefault(cls_id, []).append((score, frame_idx, box))

            # High-confidence detections as pseudo ground-truth
            if score >= PSEUDO_GT_CONF_THRESHOLD:
                pseudo_gt_by_cls.setdefault(cls_id, []).append((frame_idx, box))

        frame_idx += 1

    cap.release()
    t_infer = time.time() - t_start

    # Compute mAP over the video using pseudo ground-truth
    mAP = compute_map(pseudo_gt_by_cls, preds_by_cls, iou_threshold=IOU_THRESHOLD)

    # Second pass: run inference again to draw boxes and overlay mAP text, save video
    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # Simple deterministic color per class
    rng = np.random.RandomState(42)
    color_cache = {}

    def get_color_for_class(cid):
        if cid not in color_cache:
            color_cache[cid] = tuple(int(c) for c in rng.randint(0, 255, size=3))
        return color_cache[cid]

    cap2 = cv2.VideoCapture(input_path)
    if not cap2.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to reopen input video for rendering: {input_path}")

    # Prepare label lookup that tolerates index out of range or negative
    def class_to_label(cid):
        if 0 <= cid < len(labels):
            return labels[cid]
        return f"id_{cid}"

    overlay_text = f"mAP (pseudo-GT@{PSEUDO_GT_CONF_THRESHOLD}, IoU@{IOU_THRESHOLD}): {mAP*100:.2f}%"

    # Drawing loop
    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame, input_shape, input_dtype)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        boxes_norm, classes_raw, scores, num = parse_tflite_outputs(interpreter)
        boxes_px = convert_boxes_to_pixels(boxes_norm, width, height)

        for i in range(num):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            cid = int(classes_raw[i])
            box = boxes_px[i]
            color = get_color_for_class(cid)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = class_to_label(cid)
            text = f"{label}: {score:.2f}"
            # Text background
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(0, y1 - 10)
            cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 2, y_text + 2), color, -1)
            cv2.putText(frame, text, (x1 + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Overlay mAP text
        cv2.putText(frame, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)

        writer.write(frame)

    cap2.release()
    writer.release()

    # Optional console logs
    print(f"Processed {frame_idx} frames")
    print(f"Inference (pass 1) time: {t_infer:.2f}s, avg {t_infer / max(frame_idx, 1):.3f}s/frame")
    print(f"Output saved to: {output_path}")
    print(overlay_text)


if __name__ == "__main__":
    main()