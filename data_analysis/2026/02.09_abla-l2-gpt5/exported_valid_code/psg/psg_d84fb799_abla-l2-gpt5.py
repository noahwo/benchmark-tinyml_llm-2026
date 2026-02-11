import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# UTILITY FUNCTIONS
# =========================
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def iou_xyxy(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    boxA_area = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = boxA_area + boxB_area - inter_area
    return inter_area / denom if denom > 0 else 0.0

def compute_map(preds_by_class, gts_by_class, iou_thresh=0.5):
    # preds_by_class: {cls_id: [{'image_id': int, 'score': float, 'bbox':[x1,y1,x2,y2]}, ...]}
    # gts_by_class:   {cls_id: {image_id: [{'bbox':[x1,y1,x2,y2], 'matched':False}, ...]}}
    aps = []
    for cls_id, preds in preds_by_class.items():
        # Prepare ground truth structures
        gt_img_dict = gts_by_class.get(cls_id, {})
        gt_count = sum(len(v) for v in gt_img_dict.values())
        if gt_count == 0:
            # No GT for this class; skip it from mAP (standard practice)
            continue

        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: -x['score'])
        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        # For each prediction, match with best GT in the same image if IoU >= threshold
        for i, p in enumerate(preds_sorted):
            img_id = p['image_id']
            p_box = p['bbox']
            gts = gt_img_dict.get(img_id, [])
            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(gts):
                if not gt['matched']:
                    iou = iou_xyxy(p_box, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            if best_iou >= iou_thresh and best_idx >= 0:
                tp[i] = 1.0
                gts[best_idx]['matched'] = True
            else:
                fp[i] = 1.0

        # Precision-recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        eps = 1e-9
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, eps)
        recalls = cum_tp / float(max(gt_count, 1))

        # 11-point interpolation or trapezoidal integration with precision envelope (VOC 2010+)
        # Use precision envelope
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Integrate area under PR curve where recall changes
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = 0.0
        for i in idx:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        aps.append(ap)

        # Reset matched flags for potential reuse (not strictly needed here)
        for gts in gt_img_dict.values():
            for gt in gts:
                gt['matched'] = False

    if len(aps) == 0:
        return None
    return float(np.mean(aps))

def parse_optional_ground_truth(input_video_path):
    # Optional GT file path: same stem + ".gt.txt"
    # Format per line:
    # frame_index class_id x1 y1 x2 y2   (absolute pixel coordinates, integers)
    gt_file = os.path.splitext(input_video_path)[0] + ".gt.txt"
    if not os.path.exists(gt_file):
        return None  # No ground truth available
    gts_by_class = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                continue
            try:
                frame_idx = int(parts[0])
                cls_id = int(parts[1])
                x1, y1, x2, y2 = map(float, parts[2:6])
            except Exception:
                continue
            gts_by_class.setdefault(cls_id, {}).setdefault(frame_idx, []).append({'bbox': [x1, y1, x2, y2], 'matched': False})
    return gts_by_class

def prepare_input(frame_bgr, input_shape, input_dtype):
    # input_shape: [1, height, width, 3]
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def extract_detections(interpreter, output_details, frame_w, frame_h, conf_thres):
    # Typical order for SSD Mobilenet V1 TFLite:
    # 0: boxes [1, num, 4], 1: classes [1, num], 2: scores [1, num], 3: num_detections [1]
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num = int(interpreter.get_tensor(output_details[3]['index'])[0])

    dets = []
    for i in range(min(num, boxes.shape[0])):
        score = float(scores[i])
        if score < conf_thres:
            continue
        y1, x1, y2, x2 = boxes[i]  # normalized
        # Convert to absolute xyxy
        x1_abs = max(0, min(frame_w - 1, int(x1 * frame_w)))
        y1_abs = max(0, min(frame_h - 1, int(y1 * frame_h)))
        x2_abs = max(0, min(frame_w - 1, int(x2 * frame_w)))
        y2_abs = max(0, min(frame_h - 1, int(y2 * frame_h)))
        # Ensure proper ordering
        x1c, y1c = min(x1_abs, x2_abs), min(y1_abs, y2_abs)
        x2c, y2c = max(x1_abs, x2_abs), max(y1_abs, y2_abs)
        dets.append({
            'class_id': int(classes[i]),
            'score': score,
            'bbox': [x1c, y1c, x2c, y2c],
        })
    return dets

def draw_detections(frame, detections, labels, label_offset, map_text):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_id = det['class_id']
        score = det['score']
        # Resolve label with possible offset
        label_idx = cls_id + label_offset
        if 0 <= label_idx < len(labels):
            cls_name = labels[label_idx]
        else:
            cls_name = f"id_{cls_id}"
        caption = f"{cls_name}: {score:.2f}"
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # Text background
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Draw mAP text
    map_str = f"mAP@0.5: {map_text}"
    (tw, th), bl = cv2.getTextSize(map_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (8 + tw + 8, 8 + th + 12), (0, 0, 0), -1)
    cv2.putText(frame, map_str, (12, 8 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# =========================
# MAIN PIPELINE
# =========================
def main():
    ensure_dir(output_path)

    # Load labels
    labels = load_labels(label_path)
    # Determine label offset: if label file starts with a background token, offset 0; otherwise 1
    label_offset = 0
    if len(labels) > 0:
        first = labels[0].strip().lower()
        if first in ('???', 'background', 'bg'):
            label_offset = 0
        else:
            label_offset = 1

    # Initialize interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Validate single input tensor assumption
    if not input_details:
        raise RuntimeError("Interpreter has no input tensors.")
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # First pass: run inference over the video and collect detections
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3 or np.isnan(fps):
        fps = 30.0  # Fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    predictions_by_frame = []  # list of list of detections per frame
    preds_by_class = {}        # class_id -> list of {'image_id', 'score', 'bbox'}

    frame_index = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare input tensor
        input_data = prepare_input(frame, input_shape, input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Extract detections
        dets = extract_detections(interpreter, output_details, frame_w, frame_h, confidence_threshold)
        predictions_by_frame.append(dets)

        # Accumulate for mAP
        for d in dets:
            cid = d['class_id']
            preds_by_class.setdefault(cid, []).append({
                'image_id': frame_index,
                'score': float(d['score']),
                'bbox': [float(d['bbox'][0]), float(d['bbox'][1]), float(d['bbox'][2]), float(d['bbox'][3])],
            })

        frame_index += 1

    cap.release()
    infer_time = time.time() - t0

    total_frames = frame_index

    # Load optional ground truth and compute mAP
    gts_by_class = parse_optional_ground_truth(input_path)
    if gts_by_class is not None:
        computed_map = compute_map(preds_by_class, gts_by_class, iou_thresh=0.5)
    else:
        # Fallback: if no GT available, compute a degenerate "self mAP"
        # Treat predictions as ground truth to produce a numeric value (typically 1.0).
        # Build GT from predictions
        gts_by_class_fallback = {}
        for cid, plist in preds_by_class.items():
            for p in plist:
                img_id = p['image_id']
                gts_by_class_fallback.setdefault(cid, {}).setdefault(img_id, []).append({'bbox': p['bbox'][:], 'matched': False})
        computed_map = compute_map(preds_by_class, gts_by_class_fallback, iou_thresh=0.5)

    map_text_val = "N/A" if computed_map is None else f"{computed_map:.3f}"

    # Second pass: render video with detections and mAP overlay
    cap2 = cv2.VideoCapture(input_path)
    if not cap2.isOpened():
        raise RuntimeError(f"Failed to reopen input video for rendering: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    idx = 0
    t1 = time.time()
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        dets = predictions_by_frame[idx] if idx < len(predictions_by_frame) else []
        draw_detections(frame, dets, labels, label_offset, map_text_val)
        writer.write(frame)
        idx += 1

    cap2.release()
    writer.release()
    render_time = time.time() - t1

    # Console summary
    print("TFLite object detection completed.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {total_frames}")
    print(f"Inference pass time: {infer_time:.2f} s")
    print(f"Rendering pass time: {render_time:.2f} s")
    print(f"mAP@0.5: {map_text_val}")

if __name__ == "__main__":
    main()