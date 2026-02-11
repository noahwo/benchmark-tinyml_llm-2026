import os
import time
import numpy as np
import cv2

from ai_edge_litert.interpreter import Interpreter


# Configuration parameters
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5


def load_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name != "":
                labels.append(name)
    return labels


def get_label_name(labels, class_id):
    # Robustly map class index to label name (handling possible 0/1-based ids and "???" placeholder)
    idx = int(class_id)
    if 0 <= idx < len(labels):
        if labels[idx] != "???":
            return labels[idx]
    if 0 <= (idx - 1) < len(labels):
        if labels[idx - 1] != "???":
            return labels[idx - 1]
    return f"id{idx}"


def preprocess_frame(frame_bgr, input_details):
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    input_dtype = input_details[0]['dtype']
    resized = cv2.resize(frame_bgr, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        input_data = (rgb.astype(np.float32) / 255.0)
    else:
        # Assume quantized model expects uint8 [0,255]
        input_data = rgb.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def parse_detections(interpreter, output_details, frame_width, frame_height, score_thresh):
    # Try to find tensors by expected shapes
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    boxes = None
    classes = None
    scores = None
    num = None

    # Handle common SSD MobileNet v1 TFLite output ordering: boxes, classes, scores, num
    # Fallback by detecting shapes
    for out in outputs:
        if out.ndim == 3 and out.shape[-1] == 4:
            boxes = out  # [1, N, 4]
        elif out.ndim == 2:
            if out.shape[-1] in (10, 20, 100, 1917) or out.shape[-1] > 1:
                # Heuristic; we will match by dtype later
                pass
        elif out.size == 1:
            num = out

    # A more explicit identification by name is not guaranteed; rely on common order
    # Order: [detection_boxes (float32 [1,N,4]), detection_classes (float32 [1,N]), detection_scores (float32 [1,N]), num_detections (float32 [1])]
    if boxes is None:
        # Try identify by shape
        for out in outputs:
            if isinstance(out, np.ndarray) and out.ndim == 3 and out.shape[-1] == 4:
                boxes = out
    # Identify classes and scores by 2D arrays [1, N]
    two_d = [o for o in outputs if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[0] == 1]
    if len(two_d) >= 2:
        # Heuristic: scores are in [0,1], classes are positive integers as floats
        cand_a, cand_b = two_d[0], two_d[1]
        if np.all((cand_a >= 0) & (cand_a <= 1)):
            scores = cand_a
            classes = cand_b
        elif np.all((cand_b >= 0) & (cand_b <= 1)):
            scores = cand_b
            classes = cand_a
        else:
            # Fallback: assume order classes then scores
            classes, scores = cand_a, cand_b

    if num is None:
        # If num_detections not present, derive from length
        if boxes is not None:
            num = np.array([[boxes.shape[1]]], dtype=np.float32)
        elif scores is not None:
            num = np.array([[scores.shape[1]]], dtype=np.float32)

    if boxes is None or classes is None or scores is None or num is None:
        return []

    num = int(np.squeeze(num).astype(int))
    boxes = np.squeeze(boxes)[:num]
    classes = np.squeeze(classes)[:num]
    scores = np.squeeze(scores)[:num]

    detections = []
    for i in range(num):
        score = float(scores[i])
        if score < score_thresh:
            continue
        # boxes are in [ymin, xmin, ymax, xmax] normalized [0,1]
        y_min, x_min, y_max, x_max = boxes[i]
        x_min_i = int(max(0, min(1, x_min)) * frame_width)
        x_max_i = int(max(0, min(1, x_max)) * frame_width)
        y_min_i = int(max(0, min(1, y_min)) * frame_height)
        y_max_i = int(max(0, min(1, y_max)) * frame_height)
        detections.append({
            "bbox": [x_min_i, y_min_i, x_max_i, y_max_i],
            "class_id": int(classes[i]),
            "score": score
        })
    return detections


def draw_detections(frame, detections, labels, map_text=None):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        score = det["score"]
        label = get_label_name(labels, class_id)
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if map_text is not None:
        overlay_text = f"mAP: {map_text}"
        (tw, th), bl = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (8, 8), (8 + tw + 6, 8 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, overlay_text, (11, 8 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def iou_xyxy(box_a, box_b):
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


def compute_map(pred_by_class, gt_by_class, iou_thresh=0.5):
    # Compute per-class AP with 11-point interpolation; classes with zero GT are skipped from averaging.
    ap_list = []
    for cls_id in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
        preds = pred_by_class.get(cls_id, [])
        # preds: list of (image_id, score, bbox)
        gts = gt_by_class.get(cls_id, {})
        # gts: dict[image_id] = [bbox1, bbox2, ...]
        # Build gt matched flags
        gt_matched = {img_id: np.zeros(len(bboxes), dtype=bool) for img_id, bboxes in gts.items()}
        # Sort predictions by confidence
        preds_sorted = sorted(preds, key=lambda x: -x[1])
        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        npos = sum(len(v) for v in gts.values())
        if npos == 0:
            # No ground truth for this class, skip in AP averaging
            continue

        for i, (img_id, score, pb) in enumerate(preds_sorted):
            gt_boxes = gts.get(img_id, [])
            ovmax = 0.0
            jmax = -1
            for j, gb in enumerate(gt_boxes):
                ov = iou_xyxy(pb, gb)
                if ov > ovmax:
                    ovmax = ov
                    jmax = j
            if ovmax >= iou_thresh and jmax >= 0 and not gt_matched[img_id][jmax]:
                tp[i] = 1.0
                gt_matched[img_id][jmax] = True
            else:
                fp[i] = 1.0

        # Precision-recall
        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        recall = tp_cum / float(npos + 1e-9)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = 0.0
            if np.any(recall >= t):
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        ap_list.append(ap)

    if len(ap_list) == 0:
        # No classes with ground truth; by definition here return 0.0
        return 0.0
    return float(np.mean(ap_list))


def ensure_dir_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    # Step 1: Setup - load interpreter, labels, and open video
    labels = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir_exists(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer for: {OUTPUT_PATH}")

    # Data holders for mAP calculation (no ground-truth provided; will compute 0.0)
    preds_by_class = {}  # dict[class_id] -> list of (frame_idx, score, bbox)
    gts_by_class = {}    # dict[class_id] -> dict[frame_idx] -> list of bboxes (empty here)

    frame_idx = 0
    start_time = time.time()

    # Step 2/3/4: Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_tensor = preprocess_frame(frame, input_details)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)

        # Inference
        interpreter.invoke()

        # Parse outputs
        detections = parse_detections(
            interpreter, output_details, frame_width, frame_height, CONFIDENCE_THRESHOLD
        )

        # Accumulate predictions for mAP
        for det in detections:
            cls = det["class_id"]
            if cls not in preds_by_class:
                preds_by_class[cls] = []
            # Store bbox in [x1, y1, x2, y2] format
            preds_by_class[cls].append((frame_idx, float(det["score"]), det["bbox"]))

        # Compute a running mAP (will be 0.0 without GT)
        running_map = compute_map(preds_by_class, gts_by_class, iou_thresh=0.5)
        map_text = f"{running_map:.3f}"
        if running_map == 0.0 and len(gts_by_class) == 0:
            map_text += " (no GT)"

        # Draw detections and mAP
        annotated = draw_detections(frame.copy(), detections, labels, map_text=map_text)

        # Write frame
        writer.write(annotated)

        frame_idx += 1

    # Final mAP on all processed frames
    final_map = compute_map(preds_by_class, gts_by_class, iou_thresh=0.5)
    duration = time.time() - start_time
    print(f"Processing completed in {duration:.2f}s for {frame_idx} frames ({(frame_idx / max(duration,1e-6)):.2f} FPS).")
    if len(gts_by_class) == 0:
        print(f"Final mAP: {final_map:.3f} (no ground truth provided)")
    else:
        print(f"Final mAP: {final_map:.3f}")

    # Release resources
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()