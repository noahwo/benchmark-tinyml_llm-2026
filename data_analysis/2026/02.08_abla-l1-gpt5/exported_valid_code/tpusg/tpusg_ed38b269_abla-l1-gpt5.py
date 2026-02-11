import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =============================================================================
# Configuration (from provided parameters)
# =============================================================================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# For mAP computation (proxy without external ground truth)
IOU_THRESHOLD = 0.5
PSEUDO_GT_CONF_THRESHOLD = 0.7   # detections with score >= this are treated as pseudo ground-truth
PRED_COLLECTION_MIN_CONF = 0.05  # predictions with score >= this are collected for AP evaluation


# =============================================================================
# Utilities: labels, IoU, mAP computation
# =============================================================================
def load_labels(path):
    labels = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Support two formats:
                # "0 person" or "person"
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1].strip()
                    labels[idx] = name
                else:
                    idx = len(labels)
                    labels[idx] = line
    except Exception:
        # If labels can't be loaded, fall back to empty map
        labels = {}
    return labels


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2), pixel coords
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = float(areaA + areaB - inter)
    if denom <= 0:
        return 0.0
    return inter / denom


def compute_map_11pt(gts_by_class, preds_by_class, iou_thr=0.5):
    # VOC 2007 11-point AP; aggregate across classes with available GT
    ap_values = []

    for cls_id, gt_frames in gts_by_class.items():
        # Count total GTs
        total_gt = sum(len(lst) for lst in gt_frames.values())
        if total_gt == 0:
            continue

        preds = preds_by_class.get(cls_id, [])
        if len(preds) == 0:
            ap_values.append(0.0)
            continue

        # Sort predictions by score desc
        preds.sort(key=lambda x: x[0], reverse=True)

        # Prepare matched flags per frame
        gt_matched = {frame_idx: [False] * len(boxes) for frame_idx, boxes in gt_frames.items()}

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (score, frame_idx, pred_box) in enumerate(preds):
            gts = gt_frames.get(frame_idx, [])
            best_iou = 0.0
            best_j = -1
            for j, gt_box in enumerate(gts):
                if gt_matched[frame_idx][j]:
                    continue
                iou_val = iou(pred_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            if best_iou >= iou_thr and best_j >= 0:
                tp[i] = 1.0
                gt_matched[frame_idx][best_j] = True
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / (total_gt + 1e-8)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

        # 11-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            p = precisions[recalls >= r]
            p_max = np.max(p) if p.size > 0 else 0.0
            ap += p_max
        ap /= 11.0
        ap_values.append(ap)

    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))


# =============================================================================
# TFLite + EdgeTPU setup and inference helpers
# =============================================================================
def make_interpreter(model_path):
    return Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0")]
    )


def set_input(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    h, w = input_details['shape'][1], input_details['shape'][2]
    img_resized = cv2.resize(image, (w, h))
    if input_details['dtype'] == np.float32:
        # Normalize to [0,1]
        input_tensor = (img_resized.astype(np.float32) / 255.0).reshape(1, h, w, 3)
    else:
        input_tensor = img_resized.astype(input_details['dtype']).reshape(1, h, w, 3)
    interpreter.set_tensor(input_details['index'], input_tensor)
    return (h, w)


def get_detections(interpreter, frame_w, frame_h):
    # Typical TFLite Detection PostProcess outputs: boxes, classes, scores, count
    output_details = interpreter.get_output_details()
    tensors = [interpreter.get_tensor(od['index']) for od in output_details]

    boxes = None
    classes = None
    scores = None
    count = None

    # Identify tensors by shapes/values heuristics
    for arr in tensors:
        arr_np = np.array(arr)
        if arr_np.ndim == 3 and arr_np.shape[-1] == 4:
            boxes = arr_np[0]
        elif arr_np.ndim == 2 and arr_np.shape[0] == 1:
            vec = arr_np[0]
            # Heuristic to decide scores vs classes
            in01_ratio = np.mean((vec >= 0.0) & (vec <= 1.0))
            is_all_int_like = np.allclose(vec, np.round(vec))
            if count is None and vec.size == 1:
                count = int(vec[0])
            else:
                if is_all_int_like and in01_ratio < 0.8:
                    classes = vec.astype(np.int32)
                else:
                    # Choose the one with more values in [0,1] as scores
                    if scores is None and in01_ratio > 0.8:
                        scores = vec.astype(np.float32)

    # Fallback to typical ordering if heuristics failed
    if boxes is None or classes is None or scores is None:
        # Try by common order: boxes, classes, scores, count
        try:
            boxes = tensors[0][0]
            classes = tensors[1][0].astype(np.int32)
            scores = tensors[2][0].astype(np.float32)
            count = int(tensors[3][0]) if count is None else count
        except Exception:
            # As final fallback, attempt a best-effort scan
            for arr in tensors:
                a = np.array(arr)
                if a.ndim == 3 and a.shape[-1] == 4:
                    boxes = a[0]
            for arr in tensors:
                a = np.array(arr)
                if a.ndim == 2 and a.shape[0] == 1:
                    v = a[0]
                    if v.size == 1:
                        count = int(v[0])
                    else:
                        if np.mean((v >= 0) & (v <= 1)) > 0.8:
                            scores = v.astype(np.float32)
                        else:
                            classes = v.astype(np.int32)

    # Ensure counts are consistent
    if count is None:
        count = len(scores) if scores is not None else (len(classes) if classes is not None else 0)
    n = min(count, len(scores) if scores is not None else 0, len(classes) if classes is not None else 0, len(boxes) if boxes is not None else 0)

    if n <= 0 or boxes is None or classes is None or scores is None:
        return [], [], [], 0

    boxes = boxes[:n]
    classes = classes[:n]
    scores = scores[:n]

    # Convert normalized boxes [ymin, xmin, ymax, xmax] to pixel coords
    pixel_boxes = []
    for b in boxes:
        ymin, xmin, ymax, xmax = b
        x1 = int(max(0, min(frame_w - 1, xmin * frame_w)))
        y1 = int(max(0, min(frame_h - 1, ymin * frame_h)))
        x2 = int(max(0, min(frame_w - 1, xmax * frame_w)))
        y2 = int(max(0, min(frame_h - 1, ymax * frame_h)))
        # Ensure proper ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        pixel_boxes.append((x1, y1, x2, y2))

    return pixel_boxes, classes.tolist(), scores.tolist(), n


# =============================================================================
# Main processing
# =============================================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU delegate
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open input video:", INPUT_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print("ERROR: Cannot open output for writing:", OUTPUT_PATH)
        cap.release()
        return

    # Aggregators for mAP proxy (pseudo ground truth from high-confidence detections)
    # gts_by_class: {class_id: {frame_idx: [box, ...], ...}, ...}
    # preds_by_class: {class_id: [(score, frame_idx, box), ...], ...}
    gts_by_class = {}
    preds_by_class = {}

    frame_idx = 0
    t0 = time.time()
    running_map = 0.0

    print("Processing video...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Preprocess: BGR->RGB (TFLite models commonly use RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Set input tensor
        set_input(interpreter, frame_rgb)

        # Inference
        interpreter.invoke()

        # Collect detections
        boxes, classes, scores, num = get_detections(interpreter, frame_w, frame_h)

        # Draw detections above threshold
        for i in range(num):
            score = scores[i]
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(classes[i])
            x1, y1, x2, y2 = boxes[i]
            color = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            label_text = labels.get(cls_id, str(cls_id))
            txt = f"{label_text} {score:.2f}"
            # Text background
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty1 = max(0, y1 - th - 6)
            ty2 = y1
            tx1 = x1
            tx2 = x1 + tw + 6
            cv2.rectangle(frame_bgr, (tx1, ty1), (tx2, ty2), color, -1)
            cv2.putText(frame_bgr, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Update aggregators for mAP proxy
        # Pseudo GT: high-confidence detections
        for i in range(num):
            score = scores[i]
            cls_id = int(classes[i])
            box = boxes[i]

            # Collect pseudo GT
            if score >= PSEUDO_GT_CONF_THRESHOLD:
                if cls_id not in gts_by_class:
                    gts_by_class[cls_id] = {}
                if frame_idx not in gts_by_class[cls_id]:
                    gts_by_class[cls_id][frame_idx] = []
                gts_by_class[cls_id][frame_idx].append(box)

            # Collect predictions
            if score >= PRED_COLLECTION_MIN_CONF:
                if cls_id not in preds_by_class:
                    preds_by_class[cls_id] = []
                preds_by_class[cls_id].append((float(score), frame_idx, box))

        # Compute running mAP proxy (on frames processed so far)
        running_map = compute_map_11pt(gts_by_class, preds_by_class, IOU_THRESHOLD)

        # Overlay mAP on frame
        map_text = f"mAP (proxy): {running_map * 100.0:.2f}%"
        cv2.putText(frame_bgr, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2, cv2.LINE_AA)

        # Write frame
        writer.write(frame_bgr)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames; current mAP (proxy): {running_map:.4f}")

    cap.release()
    writer.release()
    t1 = time.time()

    print("Done.")
    print(f"Frames processed: {frame_idx}")
    print(f"Elapsed time: {t1 - t0:.2f} s")
    print(f"Final mAP (proxy): {running_map:.4f}")
    print("Output saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()