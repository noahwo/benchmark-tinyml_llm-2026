import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# ==============================
# Configuration Parameters
# ==============================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5  # for mAP computation

# ==============================
# Utilities
# ==============================

def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def make_output_dir(path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

def preprocess_frame(frame, input_shape, input_dtype, quant_params):
    # input_shape: [1, height, width, 3]
    ih, iw = input_shape[1], input_shape[2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih))
    if input_dtype == np.float32:
        # Normalize to [0,1]
        input_data = resized.astype(np.float32) / 255.0
    else:
        # Quantized uint8
        # If quantization parameters present, usually just use uint8 image data
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def run_inference(interpreter, input_data, input_index, output_details):
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Typical SSD MobileNet v1 outputs:
    # 'boxes': [1, num, 4], 'classes': [1, num], 'scores': [1, num], 'num': [1]
    boxes = interpreter.get_tensor(output_details['boxes'])[0]
    classes = interpreter.get_tensor(output_details['classes'])[0]
    scores = interpreter.get_tensor(output_details['scores'])[0]
    num = int(interpreter.get_tensor(output_details['num'])[0]) if output_details['num'] is not None else len(scores)
    return boxes, classes, scores, num

def iou(box_a, box_b):
    # boxes are [xmin, ymin, xmax, ymax]
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union

def ap_from_pr(precisions, recalls):
    # VOC-style AP computation using precision envelope
    # Add boundary points
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # Precision envelope
    for i in range(mpre.size - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    # Integrate area under curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def compute_map(preds_by_class, gts_by_class, num_gts_by_class, iou_thresh=0.5, max_frame=None):
    # If no ground truth provided, return None
    if gts_by_class is None or num_gts_by_class is None:
        return None

    aps = []
    for cid, preds in preds_by_class.items():
        # Restrict ground truths to frames up to max_frame if provided
        if cid not in gts_by_class:
            continue
        # Total GT count for this class up to max_frame
        if max_frame is None:
            gt_frames = gts_by_class[cid]
        else:
            gt_frames = {f: g for f, g in gts_by_class[cid].items() if f <= max_frame}

        npos = sum(len(v) for v in gt_frames.values())
        if npos == 0:
            continue  # skip classes with no GT so far

        # Build predictions up to max_frame
        if max_frame is None:
            preds_list = preds
        else:
            preds_list = [p for p in preds if p['frame'] <= max_frame]

        if len(preds_list) == 0:
            aps.append(0.0)
            continue

        # Sort predictions by score descending
        preds_sorted = sorted(preds_list, key=lambda x: x['score'], reverse=True)

        # Matched flags per frame for GTs
        matched = {f: np.zeros(len(gts), dtype=bool) for f, gts in gt_frames.items()}

        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        for i, pred in enumerate(preds_sorted):
            f = pred['frame']
            pb = pred['bbox']
            if f not in gt_frames:
                fp[i] = 1.0
                continue
            gts = gt_frames[f]
            if len(gts) == 0:
                fp[i] = 1.0
                continue
            # Find best IoU
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gts):
                if matched[f][j]:
                    continue
                iou_val = iou(pb, gb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0 and not matched[f][best_j]:
                tp[i] = 1.0
                matched[f][best_j] = True
            else:
                fp[i] = 1.0

        # Compute precision-recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / float(max(npos, 1))
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = ap_from_pr(precisions, recalls)
        aps.append(ap)

    if len(aps) == 0:
        return None
    return float(np.mean(aps))

def draw_detections(frame, detections, labels, map_text):
    h, w = frame.shape[:2]
    for det in detections:
        xmin, ymin, xmax, ymax = det['bbox']
        cls_id = det['class_id']
        score = det['score']
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        label = labels[cls_id] if 0 <= cls_id < len(labels) else f"id:{cls_id}"
        text = f"{label} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x1, y1 = int(xmin), int(max(0, ymin - th - 4))
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 4, y1 + th + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1 + 2, y1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw mAP text at top-left
    map_disp = f"mAP@0.5: {map_text}"
    (mw, mh), bl = cv2.getTextSize(map_disp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (5, 5), (5 + mw + 10, 5 + mh + 10), (0, 0, 0), -1)
    cv2.putText(frame, map_disp, (10, 10 + mh), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def try_load_ground_truth_txt(input_video_path, labels):
    """
    Optional ground-truth loader.
    Looks for a text file next to the video with the same base name and .txt extension.
    Expected line format (comma or whitespace separated):
        frame_index label xmin ymin xmax ymax
    Example:
        0 sheep 120 80 240 200
        0 sheep 300 100 360 190
        1 sheep 118 82 238 202
    Returns:
        gts_by_class: dict[class_id] -> dict[frame_index] -> list of [xmin, ymin, xmax, ymax]
        num_gts_by_class: dict[class_id] -> int
    If file not found or parse error, returns (None, None).
    """
    base, _ = os.path.splitext(input_video_path)
    gt_path = base + ".txt"
    if not os.path.exists(gt_path):
        return None, None

    # Build label map (lowercase matching)
    label_to_id = {lbl.lower(): i for i, lbl in enumerate(labels)}

    gts_by_class = {}
    num_gts_by_class = {}

    try:
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Replace commas with spaces, then split
                parts = line.replace(',', ' ').split()
                if len(parts) != 6:
                    # Unexpected line format, skip
                    continue
                frame_idx_s, label_s, xmin_s, ymin_s, xmax_s, ymax_s = parts
                try:
                    frame_idx = int(frame_idx_s)
                    label_lc = label_s.lower()
                    if label_lc not in label_to_id:
                        continue
                    cid = label_to_id[label_lc]
                    xmin = float(xmin_s)
                    ymin = float(ymin_s)
                    xmax = float(xmax_s)
                    ymax = float(ymax_s)
                except Exception:
                    continue

                if cid not in gts_by_class:
                    gts_by_class[cid] = {}
                    num_gts_by_class[cid] = 0
                if frame_idx not in gts_by_class[cid]:
                    gts_by_class[cid][frame_idx] = []
                gts_by_class[cid][frame_idx].append([xmin, ymin, xmax, ymax])
                num_gts_by_class[cid] += 1

        # If after parsing nothing loaded, return None
        total_gt = sum(num_gts_by_class.values()) if num_gts_by_class else 0
        if total_gt == 0:
            return None, None
        return gts_by_class, num_gts_by_class
    except Exception:
        return None, None

# ==============================
# Main Application
# ==============================

def main():
    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details_all = interpreter.get_output_details()

    # Map outputs by semantic names if possible; otherwise infer by shape
    # Try to detect standard indices for SSD models
    # Expect 4 outputs: boxes, classes, scores, num_detections
    # We'll heuristically assign based on shapes
    out_map = {'boxes': None, 'classes': None, 'scores': None, 'num': None}
    for od in output_details_all:
        shp = od.get('shape', None)
        if shp is None:
            continue
        # Shapes are typically [1, N, 4], [1, N], [1, N], [1]
        if len(shp) == 3 and shp[-1] == 4:
            out_map['boxes'] = od['index']
        elif len(shp) == 2 and shp[-1] > 1:
            # We need to distinguish classes vs scores; try dtype
            if od.get('dtype') == np.float32:
                # Could be scores or classes (classes often float in TFLite)
                # We will check name if present
                name = od.get('name', '').lower()
                if 'score' in name or 'scores' in name:
                    out_map['scores'] = od['index']
                elif 'class' in name or 'classes' in name:
                    out_map['classes'] = od['index']
                else:
                    # Fallback later if unassigned
                    pass
            else:
                # Non-float likely classes (e.g., int)
                out_map['classes'] = od['index']
        elif len(shp) == 1 and shp[0] == 1:
            out_map['num'] = od['index']

    # Final fallback if any missing (based on order convention)
    if None in out_map.values():
        # Assign by order with best guess: [boxes, classes, scores, num]
        indices = [od['index'] for od in output_details_all]
        if out_map['boxes'] is None and len(indices) > 0:
            out_map['boxes'] = indices[0]
        if out_map['classes'] is None and len(indices) > 1:
            out_map['classes'] = indices[1]
        if out_map['scores'] is None and len(indices) > 2:
            out_map['scores'] = indices[2]
        if out_map['num'] is None and len(indices) > 3:
            out_map['num'] = indices[3]

    output_details = out_map

    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    quant_params = input_details[0].get('quantization', (0.0, 0))

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video: {}".format(INPUT_PATH))

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2 or np.isnan(fps):
        fps = 25.0

    make_output_dir(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open output video writer: {}".format(OUTPUT_PATH))

    # Optional ground truth loading (text file with same base name)
    gts_by_class, num_gts_by_class = try_load_ground_truth_txt(INPUT_PATH, labels)

    # Prediction storage for mAP computation
    preds_by_class = {i: [] for i in range(len(labels))}

    frame_index = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            input_data = preprocess_frame(frame, input_shape, input_dtype, quant_params)

            # Inference
            boxes, classes, scores, num = run_inference(interpreter, input_data, input_index, output_details)

            # Collect detections above threshold and draw them
            detections = []
            for i in range(num):
                score = float(scores[i])
                if score < CONFIDENCE_THRESHOLD:
                    continue
                # Boxes are [ymin, xmin, ymax, xmax] normalized [0,1]
                ymin = float(boxes[i][0]) * in_h
                xmin = float(boxes[i][1]) * in_w
                ymax = float(boxes[i][2]) * in_h
                xmax = float(boxes[i][3]) * in_w

                # Clamp to image bounds
                xmin = max(0.0, min(xmin, in_w - 1.0))
                ymin = max(0.0, min(ymin, in_h - 1.0))
                xmax = max(0.0, min(xmax, in_w - 1.0))
                ymax = max(0.0, min(ymax, in_h - 1.0))

                # Some TFLite models return float class indices
                cls_id = int(classes[i]) if int(classes[i]) >= 0 else 0
                # Protect in case of out-of-range
                if cls_id < 0:
                    cls_id = 0

                det = {
                    'bbox': [xmin, ymin, xmax, ymax],
                    'class_id': cls_id,
                    'score': score
                }
                detections.append(det)

                # For mAP eval
                if cls_id in preds_by_class:
                    preds_by_class[cls_id].append({
                        'frame': frame_index,
                        'bbox': [xmin, ymin, xmax, ymax],
                        'score': score
                    })

            # Compute running mAP if GT available
            if gts_by_class is not None and num_gts_by_class is not None:
                current_map = compute_map(preds_by_class, gts_by_class, num_gts_by_class,
                                          iou_thresh=IOU_THRESHOLD, max_frame=frame_index)
                map_text = f"{current_map:.3f}" if current_map is not None else "N/A"
            else:
                map_text = "N/A"

            # Draw and write
            draw_detections(frame, detections, labels, map_text)
            writer.write(frame)

            frame_index += 1

    finally:
        cap.release()
        writer.release()

    t1 = time.time()
    duration = t1 - t0
    # Optional: print summary
    print("Processing completed.")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_index}")
    if duration > 0:
        print(f"Avg FPS (including I/O): {frame_index / duration:.2f}")

if __name__ == "__main__":
    main()