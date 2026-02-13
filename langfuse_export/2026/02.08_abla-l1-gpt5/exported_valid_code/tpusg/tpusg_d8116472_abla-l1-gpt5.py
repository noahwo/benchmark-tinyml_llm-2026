import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# =========================
# HELPERS
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Try "id label" format first
    parsed_any = False
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
            parsed_any = True
    if parsed_any:
        return labels
    # Fallback: one label per line, indexed from 0
    for i, line in enumerate(lines):
        labels[i] = line
    return labels

def make_interpreter(model_path, use_edgetpu=True):
    delegates = []
    if use_edgetpu:
        try:
            delegates = [load_delegate(EDGETPU_LIB)]
        except Exception as e:
            print("Warning: Failed to load EdgeTPU delegate, falling back to CPU. Error:", e)
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    return interpreter

def preprocess(frame_bgr, input_shape, input_dtype, quant_params):
    # input_shape: [1, height, width, channels]
    _, height, width, _ = input_shape
    # Convert BGR->RGB and resize to model input
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.uint8:
        # For typical EdgeTPU quant models, pass uint8 0..255 directly.
        input_data = resized.astype(np.uint8)
    else:
        # Float input: normalize to 0..1
        input_data = resized.astype(np.float32) / 255.0
        # If quantization parameters exist (rare for float inputs), ignore here.
    return np.expand_dims(input_data, axis=0)

def _tensor_by_name_or_shape(output_details):
    # Map outputs to names: boxes, classes, scores, num_detections
    # Prefer using 'name' when available, otherwise infer by shape/value ranges.
    name_map = {"boxes": None, "classes": None, "scores": None, "num": None}
    for d in output_details:
        n = d.get('name', '').lower()
        shp = d['shape']
        if 'boxes' in n:
            name_map['boxes'] = d
        elif 'classes' in n:
            name_map['classes'] = d
        elif 'scores' in n:
            name_map['scores'] = d
        elif 'num' in n:
            name_map['num'] = d

    # If names weren't informative, infer by shapes
    if not all(name_map.values()):
        # Identify boxes: shape (1, N, 4)
        for d in output_details:
            shp = d['shape']
            if len(shp) == 3 and shp[0] == 1 and shp[-1] == 4:
                name_map['boxes'] = d
                break
        # Identify num: shape (1,)
        for d in output_details:
            shp = d['shape']
            if len(shp) == 1 and shp[0] == 1:
                name_map['num'] = d
                break
        # Remaining two are classes and scores (shape (1, N))
        candidates = [d for d in output_details if d not in (name_map['boxes'], name_map['num'])]
        if len(candidates) == 2:
            # Decide which is scores by checking value range after getting tensors later
            pass
    return name_map

def extract_detections(interpreter, output_details, labels_map, frame_w, frame_h, threshold):
    # Build mapping for outputs
    name_map = _tensor_by_name_or_shape(output_details)
    tensors = {}
    # Retrieve raw tensors
    for key, d in name_map.items():
        if d is not None:
            tensors[key] = interpreter.get_tensor(d['index'])
    # If either classes or scores wasn't identified by name, infer by value range
    if 'scores' not in tensors or 'classes' not in tensors:
        # Find the two 2D outputs (1, N)
        twod = [d for d in output_details if len(d['shape']) == 2 and d['shape'][0] == 1]
        vals = [(interpreter.get_tensor(d['index']), d) for d in twod]
        # Heuristic: scores are in [0,1] and float; classes are class indices (float but typically >1)
        if 'scores' not in tensors or 'classes' not in tensors:
            # Assign scores as array with most values in [0,1]
            if len(vals) == 2:
                a0, d0 = vals[0]
                a1, d1 = vals[1]
                # Compute fraction of elements within [0,1]
                f0 = np.mean((a0 >= 0.0) & (a0 <= 1.0))
                f1 = np.mean((a1 >= 0.0) & (a1 <= 1.0))
                if f0 >= f1:
                    tensors['scores'] = a0
                    name_map['scores'] = d0
                    tensors['classes'] = a1
                    name_map['classes'] = d1
                else:
                    tensors['scores'] = a1
                    name_map['scores'] = d1
                    tensors['classes'] = a0
                    name_map['classes'] = d0

    boxes = tensors['boxes'][0] if 'boxes' in tensors else np.zeros((0, 4), dtype=np.float32)
    classes = tensors['classes'][0].astype(np.int32) if 'classes' in tensors else np.array([], dtype=np.int32)
    scores = tensors['scores'][0].astype(np.float32) if 'scores' in tensors else np.array([], dtype=np.float32)
    num = int(tensors['num'][0]) if 'num' in tensors else len(scores)

    detections = []
    for i in range(min(num, boxes.shape[0], classes.shape[0], scores.shape[0])):
        score = float(scores[i])
        if score < threshold:
            continue
        box = boxes[i]
        # Box may be normalized [ymin, xmin, ymax, xmax] or absolute.
        ymin, xmin, ymax, xmax = [float(v) for v in box]
        if max(abs(ymin), abs(xmin), abs(ymax), abs(xmax)) <= 1.5:
            # normalized
            x1 = int(max(0, min(1, xmin)) * frame_w)
            y1 = int(max(0, min(1, ymin)) * frame_h)
            x2 = int(max(0, min(1, xmax)) * frame_w)
            y2 = int(max(0, min(1, ymax)) * frame_h)
        else:
            # absolute
            x1 = int(max(0, xmin))
            y1 = int(max(0, ymin))
            x2 = int(min(frame_w - 1, xmax))
            y2 = int(min(frame_h - 1, ymax))
        if x2 <= x1 or y2 <= y1:
            continue
        cls_id = int(classes[i])
        label = labels_map.get(cls_id, str(cls_id))
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": cls_id,
            "label": label
        })
    return detections

def draw_detections(frame, detections, map_text, fps_text):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        score = det["score"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, thickness=-1)
        cv2.putText(frame, caption, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay mAP and FPS at top-left
    y0 = 20
    cv2.putText(frame, map_text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area + 1e-9
    return inter_area / union

def compute_ap_for_class(gt_boxes, pred_boxes_scores, iou_thresh=0.5):
    """
    gt_boxes: list of boxes (x1,y1,x2,y2)
    pred_boxes_scores: list of (box, score)
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        # By VOC convention, classes with no GT are ignored from mAP computation
        return None

    if len(pred_boxes_scores) == 0:
        return 0.0

    pred_sorted = sorted(pred_boxes_scores, key=lambda x: x[1], reverse=True)
    gt_matched = [False] * n_gt
    tp = np.zeros(len(pred_sorted), dtype=np.float32)
    fp = np.zeros(len(pred_sorted), dtype=np.float32)

    for i, (pbox, _) in enumerate(pred_sorted):
        # Find best IoU GT
        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_boxes):
            iou = iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0 and not gt_matched[best_j]:
            tp[i] = 1.0
            gt_matched[best_j] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / (n_gt + 1e-9)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # VOC-style AP computation (integral of precision envelope)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def compute_self_consistency_map(prev_dets, curr_dets, iou_thresh=0.5):
    """
    Approximates mAP by treating previous-frame detections as 'pseudo-GT'
    and current detections as predictions. Returns mAP across classes present
    in prev_dets.
    """
    # Group by class
    prev_by_cls = {}
    curr_by_cls = {}
    for d in prev_dets:
        prev_by_cls.setdefault(d["class_id"], []).append(d["bbox"])
    for d in curr_dets:
        curr_by_cls.setdefault(d["class_id"], []).append((d["bbox"], d["score"]))

    aps = []
    for cls_id, gt_boxes in prev_by_cls.items():
        preds = curr_by_cls.get(cls_id, [])
        ap = compute_ap_for_class(gt_boxes, preds, iou_thresh=iou_thresh)
        if ap is not None:
            aps.append(ap)
    if len(aps) == 0:
        return None
    return float(np.mean(aps))

# =========================
# MAIN PIPELINE
# =========================
def main():
    # Prepare output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU delegate
    interpreter = make_interpreter(MODEL_PATH, use_edgetpu=True)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]['index']
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_quant = input_details[0].get('quantization', (0.0, 0))

    # Video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {INPUT_PATH}")
        return

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # default fallback

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Processing loop
    frame_count = 0
    t0 = time.time()
    prev_dets = []
    map_running_sum = 0.0
    map_running_count = 0

    # For smoothed FPS
    last_time = time.time()
    fps_inst = 0.0
    fps_alpha = 0.1  # exponential smoothing

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            fh, fw = frame.shape[:2]

            # Preprocess
            input_data = preprocess(frame, input_shape, input_dtype, input_quant)
            interpreter.set_tensor(input_idx, input_data)

            # Inference
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer = time.time() - t_infer_start

            # Postprocess: parse detections and map to frame coordinates
            detections = extract_detections(interpreter, output_details, labels, fw, fh, CONFIDENCE_THRESHOLD)

            # Self-consistency mAP against previous frame detections
            map_value = compute_self_consistency_map(prev_dets, detections, iou_thresh=0.5)
            if map_value is not None:
                map_running_sum += map_value
                map_running_count += 1

            # Update instantaneous FPS
            now = time.time()
            dt = now - last_time
            last_time = now
            inst = 1.0 / dt if dt > 1e-6 else 0.0
            fps_inst = fps_alpha * inst + (1 - fps_alpha) * fps_inst

            # Overlay and write frame
            if map_running_count > 0:
                map_text = f"mAP: {map_running_sum / map_running_count:.3f}"
            else:
                map_text = "mAP: N/A"
            fps_text = f"FPS: {fps_inst:.1f} (infer {1000.0*t_infer:.1f} ms)"
            draw_detections(frame, detections, map_text, fps_text)
            writer.write(frame)

            # Prepare for next iteration
            prev_dets = detections

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    overall_fps = frame_count / total_time if total_time > 1e-6 else 0.0
    final_map = (map_running_sum / map_running_count) if map_running_count > 0 else float('nan')

    print("Processing complete.")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {overall_fps:.2f}")
    if map_running_count > 0:
        print(f"Approx. mAP (self-consistency across consecutive frames): {final_map:.4f}")
    else:
        print("Approx. mAP: N/A (insufficient detections to estimate)")

if __name__ == "__main__":
    main()