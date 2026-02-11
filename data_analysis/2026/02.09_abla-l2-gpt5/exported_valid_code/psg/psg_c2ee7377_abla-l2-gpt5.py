import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration parameters
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5  # for mAP computation

# =========================
# Utility functions
# =========================
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            labels.append(name)
    if not labels:
        labels = ["object"]
    return labels

def make_color_for_id(class_id):
    # Deterministic color from class id (BGR)
    # Spread bits to get a reasonable color distribution
    b = (37 * (class_id + 1)) % 255
    g = (17 * (class_id + 1)) % 255
    r = (29 * (class_id + 1)) % 255
    # Avoid too dark colors
    return int(b + 40) % 255, int(g + 40) % 255, int(r + 40) % 255

def preprocess(frame_bgr, input_shape, input_dtype, quant_params=None):
    # input_shape: [1, height, width, 3]
    _, ih, iw, ic = input_shape
    assert ic == 3
    # Convert BGR to RGB as most TF models expect RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.uint8:
        if quant_params and len(quant_params) == 2 and quant_params[0] > 0:
            scale, zero_point = quant_params
            # Expecting quantized input: map [0,255] -> quantized by inverse transform
            x = img_resized.astype(np.float32)
            x = x / scale + zero_point
            x = np.clip(np.round(x), 0, 255).astype(np.uint8)
        else:
            x = img_resized.astype(np.uint8)
    else:
        # Default for float models: [-1, 1] normalization used commonly by MobileNet V1
        x = img_resized.astype(np.float32)
        x = (x - 127.5) / 127.5
    x = np.expand_dims(x, axis=0)
    return x

def parse_tflite_outputs(interpreter, frame_w, frame_h, conf_thres):
    out_details = interpreter.get_output_details()
    outs = [interpreter.get_tensor(d["index"]) for d in out_details]

    # Identify standard SSD outputs: boxes [1,N,4], classes [1,N], scores [1,N], num [1]
    boxes, classes, scores, num = None, None, None, None
    for arr in outs:
        s = arr.shape
        if len(s) == 3 and s[-1] == 4:
            boxes = arr
        elif len(s) == 2:
            # Could be classes or scores
            if arr.dtype == np.float32:
                # More likely scores (float)
                if scores is None:
                    scores = arr
            else:
                # classes often float/int; fallback logic below if needed
                if classes is None:
                    classes = arr
        elif len(s) == 1 and s[0] == 1:
            num = int(np.squeeze(arr).astype(np.int32))

    # Some models return classes as float32; unify to int
    if classes is not None and classes.dtype != np.int32 and classes.dtype != np.int64:
        classes = classes.astype(np.int32)

    # Fallback: try to deduce classes/scores by dtype and values if ambiguous
    if boxes is None:
        # Try to find array with last_dim==4
        for arr in outs:
            if len(arr.shape) == 3 and arr.shape[-1] == 4:
                boxes = arr
                break

    if scores is None or classes is None:
        # Find remaining two by comparing shapes with boxes
        for arr in outs:
            if arr is boxes:
                continue
            if len(arr.shape) == 2 and arr.shape[0] == 1 and arr.shape[1] == boxes.shape[1]:
                # Distinguish by dtype/value range: scores in [0,1] float32; classes ints/floats >=0
                if arr.dtype == np.float32 and np.all((arr >= 0) & (arr <= 1)):
                    scores = arr
                else:
                    classes = arr.astype(np.int32)

    if num is None:
        num = boxes.shape[1]

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)

    detections = []
    count = min(num, boxes.shape[0])
    for i in range(count):
        score = float(scores[i])
        if score < conf_thres:
            continue
        y_min, x_min, y_max, x_max = boxes[i].tolist()
        # Convert normalized [0,1] to absolute pixel coords
        xmin = int(max(0, x_min * frame_w))
        ymin = int(max(0, y_min * frame_h))
        xmax = int(min(frame_w - 1, x_max * frame_w))
        ymax = int(min(frame_h - 1, y_max * frame_h))
        if xmax <= xmin or ymax <= ymin:
            continue
        cid = int(classes[i])
        detections.append((cid, score, (xmin, ymin, xmax, ymax)))
    return detections

def draw_detections(frame, detections, labels, map_text):
    # Draw mAP and threshold info header
    h, w = frame.shape[:2]
    y0 = 22
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), thickness=-1)
    header = f"Confidence >= {CONF_THRESHOLD:.2f}   {map_text}"
    cv2.putText(frame, header, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    for cid, score, (x1, y1, x2, y2) in detections:
        color = make_color_for_id(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = labels[cid] if 0 <= cid < len(labels) else f"id:{cid}"
        caption = f"{label}: {score:.2f}"
        # Text background
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, ty1), (x1 + tw + 4, ty1 + th + 6), color, thickness=-1)
        cv2.putText(frame, caption, (x1 + 2, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def voc_ap(rec, prec):
    # Compute AP by integrating precision envelope (VOC style)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def compute_map(preds_per_frame, gts_per_frame, num_classes, iou_th=0.5):
    # Prepare GT structures
    # gts_per_frame: list of list of tuples (cid, bbox)
    # preds_per_frame: list of list of tuples (cid, score, bbox)
    gt_count_per_class = {c: 0 for c in range(num_classes)}
    gt_by_class_image = {c: {} for c in range(num_classes)}  # c -> img_id -> list of (bbox, matched_flag)
    for img_id, gts in enumerate(gts_per_frame):
        for cid, bbox in gts:
            if cid not in gt_by_class_image:
                gt_by_class_image[cid] = {}
            if img_id not in gt_by_class_image[cid]:
                gt_by_class_image[cid][img_id] = []
            gt_by_class_image[cid][img_id].append([bbox, False])  # [bbox, matched]
            gt_count_per_class[cid] = gt_count_per_class.get(cid, 0) + 1

    aps = []
    for c in range(num_classes):
        # Collect predictions for this class
        preds = []
        for img_id, dets in enumerate(preds_per_frame):
            for cid, score, bbox in dets:
                if cid == c:
                    preds.append((img_id, score, bbox))
        if len(preds) == 0:
            # No predictions for this class
            if gt_count_per_class.get(c, 0) > 0:
                aps.append(0.0)
            continue
        # Sort by descending score
        preds.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        total_gts = gt_count_per_class.get(c, 0)
        if total_gts == 0:
            # No GT for this class; ignore in mAP
            continue

        for i, (img_id, score, pb) in enumerate(preds):
            matched = False
            candidates = gt_by_class_image.get(c, {}).get(img_id, [])
            best_iou = 0.0
            best_j = -1
            for j, (gb, used) in enumerate(candidates):
                if used:
                    continue
                iou = iou_xyxy(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_th and best_j >= 0:
                # Match to this GT
                candidates[best_j][1] = True
                matched = True
            if matched:
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        # Precision-recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / float(total_gts + 1e-8)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
        ap = voc_ap(rec, prec)
        aps.append(ap)

    if len(aps) == 0:
        return None  # No GT or no valid classes -> mAP not applicable
    return float(np.mean(aps))

def try_load_ground_truths(input_path, frame_count):
    """
    Attempt to load ground-truth boxes for mAP from a sidecar file.
    Expected formats (CSV or space separated), one entry per line:
        frame_index, class_id, xmin, ymin, xmax, ymax
    Search order:
        1) <input_stem>_gt.txt
        2) <input_dir>/<input_filename>.gt.txt
    Returns: list of per-frame lists of tuples (class_id, (xmin, ymin, xmax, ymax))
    """
    candidates = []
    base, ext = os.path.splitext(input_path)
    candidates.append(base + "_gt.txt")
    candidates.append(input_path + ".gt.txt")

    gt_file = None
    for p in candidates:
        if os.path.isfile(p):
            gt_file = p
            break

    gts_per_frame = [[] for _ in range(frame_count)]
    if gt_file is None:
        return gts_per_frame, False

    try:
        with open(gt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Accept commas or spaces
                parts = [p for p in line.replace(",", " ").split(" ") if p != ""]
                if len(parts) < 6:
                    continue
                fi = int(parts[0])
                cid = int(parts[1])
                xmin = int(float(parts[2])); ymin = int(float(parts[3]))
                xmax = int(float(parts[4])); ymax = int(float(parts[5]))
                if 0 <= fi < frame_count:
                    gts_per_frame[fi].append((cid, (xmin, ymin, xmax, ymax)))
    except Exception:
        # If parsing fails, fall back to no GT
        return [[] for _ in range(frame_count)], False

    return gts_per_frame, True

# =========================
# Main pipeline
# =========================
def main():
    ensure_dir(OUTPUT_PATH)
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    quant_params = None
    if "quantization" in input_details[0] and input_details[0]["quantization"] is not None:
        q = input_details[0]["quantization"]
        # Some interpreters provide (scale, zero_point) tuple; ensure sanity
        if isinstance(q, tuple) and len(q) == 2:
            quant_params = q

    # Pass 1: Run inference on all frames and store detections
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # For some codecs CAP_PROP_FRAME_COUNT may be unreliable; we will count frames if needed
    frame_indices = []
    preds_per_frame = []

    t0 = time.time()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x = preprocess(frame, input_shape, input_dtype, quant_params)
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        detections = parse_tflite_outputs(interpreter, src_w, src_h, CONF_THRESHOLD)

        frame_indices.append(frame_idx)
        preds_per_frame.append(detections)
        frame_idx += 1
    cap.release()
    actual_frame_count = len(preds_per_frame)
    if total_frames <= 0:
        total_frames = actual_frame_count

    # Try load ground-truths for mAP
    gts_per_frame, gt_available = try_load_ground_truths(INPUT_PATH, actual_frame_count)

    # Compute mAP
    num_classes = max(len(labels), max([cid for dets in preds_per_frame for (cid, _, _) in dets], default=-1) + 1)
    mAP_value = compute_map(preds_per_frame, gts_per_frame, num_classes, iou_th=IOU_THRESHOLD) if gt_available else None

    # Pass 2: Re-read video and write output with overlays and mAP text
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to reopen input video for writing: {INPUT_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (src_w, src_h), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    if mAP_value is None:
        map_text = "mAP@0.50: N/A (no GT)"
    else:
        map_text = f"mAP@0.50: {mAP_value:.3f}"

    write_idx = 0
    while write_idx < actual_frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        detections = preds_per_frame[write_idx]
        draw_detections(frame, detections, labels, map_text)
        writer.write(frame)
        write_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    print(f"Processed {actual_frame_count} frames in {elapsed:.2f}s "
          f"({(actual_frame_count / max(elapsed, 1e-6)):.2f} FPS).")
    print(f"Output saved to: {OUTPUT_PATH}")
    if mAP_value is None:
        print("mAP not computed (no ground-truth file found). To enable mAP, create a GT file with lines: "
              "'frame_index, class_id, xmin, ymin, xmax, ymax' at either '<input>_gt.txt' or '<input>.gt.txt'.")
    else:
        print(f"Computed mAP@0.50: {mAP_value:.4f}")

if __name__ == "__main__":
    main()