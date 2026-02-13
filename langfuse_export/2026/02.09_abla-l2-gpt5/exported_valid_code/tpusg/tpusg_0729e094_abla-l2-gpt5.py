import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate


def load_labels(label_path):
    labels = {}
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Try common formats:
            # 1) "0 person"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
                continue
            # 2) "id:name" or "id, name"
            sep_idx = None
            for sep in [':', ',', ';']:
                if sep in line:
                    sep_idx = line.find(sep)
                    break
            if sep_idx is not None:
                left = line[:sep_idx].strip()
                right = line[sep_idx+1:].strip()
                if left.isdigit():
                    labels[int(left)] = right
                    continue
            # 3) Fallback: line number as id
            labels[i] = line
    return labels


def make_interpreter(model_path, edgetpu_lib_path):
    delegates = []
    if os.path.exists(edgetpu_lib_path):
        try:
            delegates.append(load_delegate(edgetpu_lib_path))
        except Exception as e:
            print("Warning: Failed to load EdgeTPU delegate:", e)
    else:
        print("Warning: EdgeTPU shared library not found at:", edgetpu_lib_path)
    return Interpreter(model_path=model_path, experimental_delegates=delegates)


def preprocess_frame(frame_bgr, input_size, input_dtype):
    ih, iw = input_size
    # Convert BGR (OpenCV) to RGB (common for TFLite models)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = (resized.astype(np.float32) / 255.0).astype(np.float32)
    else:
        # Assume quantized uint8
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def get_output_tensors(interpreter):
    details = interpreter.get_output_details()
    tensors = [interpreter.get_tensor(d['index']) for d in details]
    # Heuristic to assign outputs: boxes, classes, scores, count
    boxes = None
    classes = None
    scores = None
    count = None
    # Try common SSD order
    if len(tensors) >= 3:
        t0, t1, t2 = tensors[0], tensors[1], tensors[2]
        # Typical: (1,N,4), (1,N), (1,N), (1,)
        if t0.ndim == 3 and t0.shape[-1] == 4:
            boxes = t0[0]
            if t1.ndim >= 2:
                classes = t1[0].astype(np.int32)
            if t2.ndim >= 2:
                scores = t2[0].astype(np.float32)
            if len(tensors) >= 4 and tensors[3].size >= 1:
                count = int(tensors[3].flatten()[0])
    # Fallback scan
    if boxes is None or classes is None or scores is None:
        for t in tensors:
            ts = t.squeeze()
            if ts.ndim == 2 and ts.shape[-1] == 4 and boxes is None:
                boxes = ts
            elif ts.ndim == 1 and np.issubdtype(ts.dtype, np.floating) and scores is None:
                scores = ts.astype(np.float32)
            elif ts.ndim == 1 and np.issubdtype(ts.dtype, np.integer) and classes is None:
                classes = ts.astype(np.int32)
            elif ts.ndim == 0 and count is None:
                count = int(ts)
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((boxes.shape[0],), dtype=np.int32)
    if scores is None:
        scores = np.zeros((boxes.shape[0],), dtype=np.float32)
    if count is None:
        count = min(len(scores), len(boxes))
    return boxes, classes, scores, count


def detect_objects(interpreter, frame_bgr, threshold):
    input_details = interpreter.get_input_details()[0]
    _, ih, iw, _ = input_details['shape']
    input_dtype = input_details['dtype']
    input_data = preprocess_frame(frame_bgr, (ih, iw), input_dtype)

    interpreter.set_tensor(input_details['index'], input_data)

    t0 = time.time()
    interpreter.invoke()
    inference_ms = (time.time() - t0) * 1000.0

    boxes, classes, scores, count = get_output_tensors(interpreter)
    H, W = frame_bgr.shape[:2]
    detections = []
    for i in range(int(count)):
        score = float(scores[i])
        if score < threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        # Boxes are typically normalized [0,1]
        x1 = max(0, min(W - 1, int(x_min * W)))
        y1 = max(0, min(H - 1, int(y_min * H)))
        x2 = max(0, min(W - 1, int(x_max * W)))
        y2 = max(0, min(H - 1, int(y_max * H)))
        # Ensure proper ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'score': score,
            'class_id': int(classes[i])
        })
    return detections, inference_ms


def draw_detections(frame_bgr, detections, labels, color_map):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        score = det['score']
        color = color_map.get(class_id, (0, 255, 0))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = labels.get(class_id, str(class_id))
        caption = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_bgr, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def compute_map(predictions_by_frame, ground_truth_by_frame, iou_thresh=0.5):
    # predictions_by_frame: list indexed by frame_idx -> list of {'bbox','class_id','score'}
    # ground_truth_by_frame: dict frame_idx -> list of {'bbox','class_id'}
    # If no ground truth available, return None
    if not ground_truth_by_frame:
        return None

    # Gather classes from GT
    classes = set()
    for gts in ground_truth_by_frame.values():
        for gt in gts:
            classes.add(gt['class_id'])
    if not classes:
        return None

    ap_list = []
    for cls in sorted(list(classes)):
        # Collect predictions for this class
        preds = []
        gts_by_frame = {}
        for fi, preds_list in enumerate(predictions_by_frame):
            cls_preds = [p for p in preds_list if p['class_id'] == cls]
            for p in cls_preds:
                preds.append((fi, p['bbox'], p['score']))
        # Sort predictions by score descending
        preds.sort(key=lambda x: x[2], reverse=True)

        # Collect GTs for this class
        total_gts = 0
        for fi, gts in ground_truth_by_frame.items():
            cls_gts = [g for g in gts if g['class_id'] == cls]
            total_gts += len(cls_gts)
            if cls_gts:
                gts_by_frame[fi] = {
                    'boxes': [g['bbox'] for g in cls_gts],
                    'matched': [False] * len(cls_gts)
                }
        if total_gts == 0:
            # No GT for this class; skip in mAP calculation
            continue

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for i, (fi, pbox, _) in enumerate(preds):
            if fi not in gts_by_frame:
                fp[i] = 1.0
                continue
            gt_entry = gts_by_frame[fi]
            ious = [compute_iou(pbox, gbox) for gbox in gt_entry['boxes']]
            if not ious:
                fp[i] = 1.0
                continue
            max_iou = max(ious)
            max_idx = int(np.argmax(ious))
            if max_iou >= iou_thresh and not gt_entry['matched'][max_idx]:
                tp[i] = 1.0
                gt_entry['matched'][max_idx] = True
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (total_gts + 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

        # 11-point interpolated AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = 0.0
            if len(precisions) > 0:
                p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
            ap += p / 11.0
        ap_list.append(ap)

    if not ap_list:
        return None
    return float(np.mean(ap_list))


def ensure_dir_exists(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    # CONFIGURATION PARAMETERS
    model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
    label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
    input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
    output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    # Step 1: Setup
    labels = load_labels(label_path)
    interpreter = make_interpreter(model_path, "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0")
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Unable to open input video:", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ensure_dir_exists(output_path)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: Unable to open output video for writing:", output_path)
        cap.release()
        return

    # Deterministic color map per class id
    rng = np.random.RandomState(42)
    color_map = {}
    for cid in range(0, 256):
        color_map[cid] = tuple(int(c) for c in rng.randint(0, 255, size=3))

    predictions_by_frame = []
    # Optional: ground truth annotations not provided in configuration.
    ground_truth_by_frame = {}  # frame_index -> list of {'bbox':(x1,y1,x2,y2), 'class_id':int}

    frame_index = 0
    avg_inference_ms = 0.0
    t_start = time.time()

    # Step 2-4: Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, infer_ms = detect_objects(interpreter, frame, confidence_threshold)
        avg_inference_ms += infer_ms
        predictions_by_frame.append(detections)

        draw_detections(frame, detections, labels, color_map)

        # Compute mAP if ground truth is available; else N/A
        current_map = compute_map(predictions_by_frame, ground_truth_by_frame, iou_thresh=0.5)
        map_text = "mAP: N/A" if current_map is None else f"mAP: {current_map:.3f}"

        elapsed = time.time() - t_start
        fps_running = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
        overlay = f"Infer: {infer_ms:.1f} ms | FPS: {fps_running:.1f} | {map_text}"
        cv2.putText(frame, overlay, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 10), 1, cv2.LINE_AA)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    if frame_index > 0:
        avg_inference_ms /= frame_index
    final_map = compute_map(predictions_by_frame, ground_truth_by_frame, iou_thresh=0.5)
    print("Processing complete.")
    print(f"Frames processed: {frame_index}")
    print(f"Average inference time: {avg_inference_ms:.2f} ms")
    if final_map is None:
        print("mAP: N/A (no ground truth provided)")
    else:
        print(f"mAP@0.5IoU: {final_map:.4f}")
    print("Saved output video to:", output_path)


if __name__ == "__main__":
    main()