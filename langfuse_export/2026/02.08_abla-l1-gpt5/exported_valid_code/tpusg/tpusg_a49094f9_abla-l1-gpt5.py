import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =======================
# Configuration parameters
# =======================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5  # Low threshold for predictions

# Pseudo-GT (for mAP estimation without real ground truth)
high_conf_as_gt = 0.8       # Detections with score >= this are treated as pseudo-GT
iou_threshold = 0.5         # IoU threshold for matching (mAP@0.5)


# =======================
# Utilities
# =======================
def load_labels(path):
    labels = {}
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                else:
                    labels[i] = line
    except Exception as e:
        print(f"Warning: Unable to load labels from {path}: {e}")
    return labels


def make_interpreter(model_file):
    try:
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        print(f"EdgeTPU delegate load failed ({e}). Falling back to CPU.")
        interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame_bgr, input_shape, input_dtype):
    ih, iw = input_shape[1], input_shape[2]  # [1, H, W, C]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    return np.expand_dims(input_data, axis=0)


def parse_tflite_outputs(interpreter):
    # Most TPU SSD models: [boxes, classes, scores, num_detections]
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(d['index']) for d in output_details]

    # Try default order; handle shape-based fallback if needed
    boxes, classes, scores, num = None, None, None, None
    # Identify boxes (last dim 4)
    for i, out in enumerate(outputs):
        if out.ndim == 3 and out.shape[-1] == 4:
            boxes = out
            break
    # Identify num_detections (scalar or [1])
    for i, out in enumerate(outputs):
        if out.size == 1 and out.ndim <= 2:
            num = int(np.squeeze(out).astype(np.int32))
            break
    # Remaining two are classes and scores (shape [1, N])
    candidates = []
    for out in outputs:
        if out is boxes or (num is not None and out.size == 1 and out.ndim <= 2):
            continue
        if out.ndim == 2:  # [1, N]
            candidates.append(out)
    # Heuristic: scores are typically float in [0,1], classes are integer-like floats
    if len(candidates) == 2:
        a = np.squeeze(candidates[0]).astype(np.float32)
        b = np.squeeze(candidates[1]).astype(np.float32)
        if (a.max() <= 1.0 and a.min() >= 0.0) and not (b.max() <= 1.0 and b.min() >= 0.0):
            scores = candidates[0]
            classes = candidates[1]
        elif (b.max() <= 1.0 and b.min() >= 0.0) and not (a.max() <= 1.0 and a.min() >= 0.0):
            scores = candidates[1]
            classes = candidates[0]
        else:
            # Fallback to typical order [boxes, classes, scores, num]
            classes = candidates[0]
            scores = candidates[1]

    boxes = np.squeeze(boxes) if boxes is not None else None
    classes = np.squeeze(classes).astype(np.int32) if classes is not None else None
    scores = np.squeeze(scores).astype(np.float32) if scores is not None else None
    num = int(num) if num is not None else (len(scores) if scores is not None else 0)

    # Truncate to num
    if boxes is not None and len(boxes.shape) > 1:
        boxes = boxes[:num]
    if classes is not None:
        classes = classes[:num]
    if scores is not None:
        scores = scores[:num]

    return boxes, classes, scores, num


def iou(box_a, box_b):
    # boxes in [ymin, xmin, ymax, xmax] normalized or absolute (same scale)
    y1, x1, y2, x2 = box_a
    y1b, x1b, y2b, x2b = box_b
    inter_y1 = max(y1, y1b)
    inter_x1 = max(x1, x1b)
    inter_y2 = min(y2, y2b)
    inter_x2 = min(x2, x2b)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (y2 - y1)) * max(0.0, (x2 - x1))
    area_b = max(0.0, (y2b - y1b)) * max(0.0, (x2b - x1b))
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def update_metrics_for_frame(boxes_norm, classes, scores, metrics):
    # Build per-class GTs (pseudo) and predictions for this frame
    gt_by_class = {}
    pred_by_class = {}

    for b, c, s in zip(boxes_norm, classes, scores):
        if s >= high_conf_as_gt:
            gt_by_class.setdefault(c, []).append(b.tolist())
        if s >= confidence_threshold:
            pred_by_class.setdefault(c, []).append((float(s), b.tolist()))

    # Ensure class entries exist
    for c in set(list(gt_by_class.keys()) + list(pred_by_class.keys())):
        if c not in metrics['per_class']:
            metrics['per_class'][c] = {'scores': [], 'tps': [], 'gts': 0}

    # Match predictions to GT (greedy by score)
    for c, preds in pred_by_class.items():
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        gts = gt_by_class.get(c, [])
        gt_used = [False] * len(gts)

        for score, pbox in preds_sorted:
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gtbox in enumerate(gts):
                if gt_used[gi]:
                    continue
                i = iou(pbox, gtbox)
                if i > best_iou:
                    best_iou = i
                    best_gt_idx = gi
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_used[best_gt_idx] = True
                metrics['per_class'][c]['scores'].append(score)
                metrics['per_class'][c]['tps'].append(1)
            else:
                metrics['per_class'][c]['scores'].append(score)
                metrics['per_class'][c]['tps'].append(0)

    # Accumulate GT counts
    for c, gts in gt_by_class.items():
        metrics['per_class'][c]['gts'] += len(gts)


def compute_map(metrics):
    aps = []
    for c, info in metrics['per_class'].items():
        gts = info['gts']
        if gts <= 0:
            continue
        scores = np.array(info['scores'], dtype=np.float32)
        tps = np.array(info['tps'], dtype=np.int32)

        if scores.size == 0:
            aps.append(0.0)
            continue

        # Sort by score descending
        order = np.argsort(-scores)
        tps_sorted = tps[order]
        fps_sorted = 1 - tps_sorted

        cum_tp = np.cumsum(tps_sorted).astype(np.float32)
        cum_fp = np.cumsum(fps_sorted).astype(np.float32)

        recall = cum_tp / float(gts + 1e-9)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        # VOC-style AP (monotonic precision envelope)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        aps.append(float(ap))

    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))


def draw_detections(frame_bgr, boxes_norm, classes, scores, labels, thresh):
    h, w = frame_bgr.shape[:2]
    for b, c, s in zip(boxes_norm, classes, scores):
        if s < thresh:
            continue
        ymin, xmin, ymax, xmax = b
        x1 = max(0, int(xmin * w))
        y1 = max(0, int(ymin * h))
        x2 = min(w - 1, int(xmax * w))
        y2 = min(h - 1, int(ymax * h))
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = labels.get(int(c), f"id:{int(c)}")
        text = f"{label} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    # Setup
    labels = load_labels(label_path)
    interpreter = make_interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Cannot open output writer: {output_path}")
        cap.release()
        return

    # Metrics accumulator for pseudo mAP
    metrics = {'per_class': {}, 'iou_thresh': iou_threshold}

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            input_data = preprocess_frame(frame, input_shape, input_dtype)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Outputs
            boxes, classes, scores, num = parse_tflite_outputs(interpreter)
            if boxes is None or classes is None or scores is None:
                # If parsing failed, write original frame and continue
                writer.write(frame)
                frame_idx += 1
                continue

            # Update metrics using normalized boxes
            update_metrics_for_frame(boxes, classes, scores, metrics)
            mAP_val = compute_map(metrics)

            # Draw detections on frame
            draw_detections(frame, boxes, classes, scores, labels, confidence_threshold)

            # Draw mAP on frame
            map_text = f"mAP@0.5 (pseudo-GT≥{high_conf_as_gt:.1f}): {mAP_val:.3f}"
            (tw, th), _ = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (8, 8), (8 + tw + 8, 8 + th + 12), (0, 0, 0), -1)
            cv2.putText(frame, map_text, (12, 8 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        dt = time.time() - t0
        if frame_idx > 0:
            print(f"Processed {frame_idx} frames in {dt:.2f}s ({frame_idx / max(dt,1e-6):.2f} FPS)")
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()