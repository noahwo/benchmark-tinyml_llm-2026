import os
import time
import numpy as np
import cv2

from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    labels = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    except Exception:
        pass
    if not labels:
        labels = ["unknown"]
    return labels

def preprocess(frame, input_size, input_dtype):
    h, w = input_size
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)
    return img

def select_output_tensors(interpreter):
    out_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(d['index']) for d in out_details]

    boxes = None
    classes = None
    scores = None
    num_dets = None

    for arr in outputs:
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 4:
            boxes = arr
    for arr in outputs:
        if arr.size == 1:
            num_dets = arr
    # Scores typically in [0,1]
    for arr in outputs:
        if arr.ndim == 2 and arr.shape[0] == 1 and arr.dtype == np.float32:
            maxv = float(np.max(arr)) if arr.size > 0 else 0.0
            if maxv <= 1.00001:
                scores = arr
    # Classes typically > 1.0 floats (indices)
    for arr in outputs:
        if arr.ndim == 2 and arr.shape[0] == 1 and arr.dtype == np.float32:
            maxv = float(np.max(arr)) if arr.size > 0 else 0.0
            if maxv > 1.0:
                classes = arr

    return boxes, classes, scores, num_dets

def draw_labelled_box(img, x1, y1, x2, y2, label, score, color=(0, 255, 0)):
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    caption = f"{label}: {score:.2f}"
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def compute_ap(scores, tps, fps, num_gt):
    if num_gt <= 0 or len(scores) == 0:
        return 0.0
    scores = np.asarray(scores, dtype=np.float32)
    tps = np.asarray(tps, dtype=np.float32)
    fps = np.asarray(fps, dtype=np.float32)
    order = np.argsort(-scores)
    tps = tps[order]
    fps = fps[order]
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    rec = cum_tp / float(num_gt)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
    return ap

def overlay_metrics(frame, text, pos=(10, 30), color=(255, 255, 255), bg=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 6), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def main():
    labels = load_labels(label_path)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    input_index = in_details[0]['index']
    input_h, input_w = in_details[0]['shape'][1], in_details[0]['shape'][2]
    input_dtype = in_details[0]['dtype']

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Unable to open input video:", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Error: Unable to open output writer:", output_path)
        cap.release()
        return

    # Metrics accumulators (heuristic, without external GT)
    # For each class id: track scores, TP/FP flags, and approximated num_gt (at most 1 per frame per class)
    metrics = {}
    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        input_tensor = preprocess(frame, (input_h, input_w), input_dtype)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

        # Retrieve outputs and identify tensors
        boxes, classes, scores, num_dets = select_output_tensors(interpreter)
        if boxes is None or classes is None or scores is None:
            # Fallback: try to read via out_details indices if select failed
            outs = [interpreter.get_tensor(d['index']) for d in out_details]
            if len(outs) >= 3:
                boxes, classes, scores = outs[0], outs[1], outs[2]
            else:
                writer.write(frame)
                continue

        boxes = np.squeeze(boxes, axis=0)  # [N,4] normalized
        classes = np.squeeze(classes, axis=0)  # [N]
        scores = np.squeeze(scores, axis=0)  # [N]
        if num_dets is not None:
            n = int(np.squeeze(num_dets))
            boxes = boxes[:n]
            classes = classes[:n]
            scores = scores[:n]

        # Filter detections by confidence threshold
        keep = scores >= confidence_threshold
        boxes_f = boxes[keep]
        classes_f = classes[keep]
        scores_f = scores[keep]

        # Group detections by class to update heuristic mAP metrics
        per_class_indices = {}
        for i, cid in enumerate(classes_f):
            c = int(cid)
            per_class_indices.setdefault(c, []).append(i)

        for c, idxs in per_class_indices.items():
            # Ensure data structure
            if c not in metrics:
                metrics[c] = {"scores": [], "tp": [], "fp": [], "num_gt": 0}
            # Sort detections of this class in current frame by score
            order = sorted(idxs, key=lambda k: float(scores_f[k]), reverse=True)
            if len(order) > 0:
                # Heuristic: assume at most one true object per class per frame
                metrics[c]["num_gt"] += 1
                # First is TP, rest are FP
                for j, k in enumerate(order):
                    sc = float(scores_f[k])
                    if j == 0:
                        metrics[c]["scores"].append(sc)
                        metrics[c]["tp"].append(1.0)
                        metrics[c]["fp"].append(0.0)
                    else:
                        metrics[c]["scores"].append(sc)
                        metrics[c]["tp"].append(0.0)
                        metrics[c]["fp"].append(1.0)

        # Draw detections
        for i in range(len(boxes_f)):
            y1, x1, y2, x2 = boxes_f[i]
            x1p = int(x1 * width)
            y1p = int(y1 * height)
            x2p = int(x2 * width)
            y2p = int(y2 * height)
            cid = int(classes_f[i])
            label = labels[cid] if 0 <= cid < len(labels) else f"id_{cid}"
            draw_labelled_box(frame, x1p, y1p, x2p, y2p, label, float(scores_f[i]))

        # Compute mAP (heuristic) across classes that have any approximated GT so far
        aps = []
        for c, data in metrics.items():
            ap_c = compute_ap(data["scores"], data["tp"], data["fp"], data["num_gt"])
            aps.append(ap_c)
        mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0

        overlay_metrics(frame, f"mAP (heuristic): {mAP:.3f}", pos=(10, 30), color=(255, 255, 255), bg=(0, 0, 0))

        writer.write(frame)

    elapsed = time.time() - t_start
    cap.release()
    writer.release()

    # Final reporting
    final_aps = []
    for c, data in metrics.items():
        ap_c = compute_ap(data["scores"], data["tp"], data["fp"], data["num_gt"])
        final_aps.append(ap_c)
    final_mAP = float(np.mean(final_aps)) if len(final_aps) > 0 else 0.0

    print(f"Processed {frame_count} frames in {elapsed:.2f}s ({(frame_count / max(elapsed,1e-6)):.2f} FPS).")
    print(f"Saved annotated video with detections and mAP overlay to: {output_path}")
    print(f"Final mAP (heuristic, no external ground truth): {final_mAP:.4f}")

if __name__ == "__main__":
    main()