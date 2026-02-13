import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# Utility Functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        print(f"Warning: label file not found at {path}")
        return labels
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Try "index label" format; fallback to line index
    for i, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
        else:
            labels[i] = line
    return labels

def set_input_tensor(interpreter, input_data):
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, input_data)

def get_output_tensors(interpreter):
    """
    Returns (boxes, classes, scores, num) as numpy arrays.
    boxes: [N, 4] in ymin, xmin, ymax, xmax normalized coords
    classes: [N] float class indices
    scores: [N] float confidences
    num: scalar number of valid detections (may be absent on some models; fallback to len(scores))
    """
    output_details = interpreter.get_output_details()
    boxes = classes = scores = num = None

    # Try by name first
    name_map = {d['name']: d for d in output_details if 'name' in d}
    def find_by_name(substr):
        for name, det in name_map.items():
            if substr in name:
                return det
        return None

    det_boxes = find_by_name('TFLite_Detection_PostProcess') or find_by_name('boxes')
    det_scores = find_by_name('scores')
    det_classes = find_by_name('classes')
    det_num = find_by_name('num_detections')

    # Fallback by heuristics if needed
    if det_boxes is None or det_scores is None or det_classes is None:
        # Heuristic: look for shapes
        for d in output_details:
            shape = d['shape']
            if len(shape) == 3 and shape[-1] == 4:
                det_boxes = d
            elif len(shape) == 2:
                # Could be scores or classes; we will differentiate by dtype
                # Scores are float, classes may be float too; we'll decide after reading
                pass
            elif len(shape) == 1 and shape[0] == 1:
                det_num = d

        # If still ambiguous, assume typical order [boxes, classes, scores, num]
        if det_boxes is None or det_classes is None or det_scores is None:
            if len(output_details) >= 3:
                det_boxes = output_details[0]
                det_classes = output_details[1]
                det_scores = output_details[2]
            if len(output_details) >= 4:
                det_num = output_details[3]

    # Read tensors
    def tensor(d):
        return interpreter.get_tensor(d['index']) if d is not None else None

    boxes = tensor(det_boxes)
    classes = tensor(det_classes)
    scores = tensor(det_scores)
    num = tensor(det_num) if det_num is not None else None

    # Squeeze to expected shapes
    if boxes is not None and boxes.ndim == 3:
        boxes = boxes[0]
    if classes is not None and classes.ndim == 2:
        classes = classes[0]
    if scores is not None and scores.ndim == 2:
        scores = scores[0]
    if num is not None:
        num = int(np.squeeze(num).astype(np.int32))
    else:
        num = len(scores) if scores is not None else 0

    return boxes, classes, scores, num

def preprocess_frame(frame, input_size, input_dtype):
    ih, iw = input_size
    resized = cv2.resize(frame, (iw, ih))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.uint8:
        input_data = np.expand_dims(rgb, axis=0)
    else:
        input_data = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
    return input_data

def denormalize_box(box, img_w, img_h):
    ymin, xmin, ymax, xmax = box
    x1 = max(0, min(int(xmin * img_w), img_w - 1))
    y1 = max(0, min(int(ymin * img_h), img_h - 1))
    x2 = max(0, min(int(xmax * img_w), img_w - 1))
    y2 = max(0, min(int(ymax * img_h), img_h - 1))
    return [x1, y1, x2, y2]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    indices = scores_np.argsort()[::-1]
    selected = []
    while indices.size > 0:
        current = indices[0]
        selected.append(current)
        if indices.size == 1:
            break
        rest = indices[1:]
        ious = np.array([iou(boxes_np[current], boxes_np[i]) for i in rest])
        keep = np.where(ious <= iou_threshold)[0]
        indices = rest[keep]
    return selected

def compute_map_pseudogt(detections_per_frame, img_size, conf_threshold=0.5, iou_thresh=0.5):
    """
    detections_per_frame: list of list of dicts per frame
        each detection dict: {'box':[x1,y1,x2,y2], 'score':float, 'class_id':int}
    Pseudo-GT construction:
        - For each frame and class, apply NMS on detections with score >= conf_threshold.
    Evaluation:
        - VOC 2007 11-point AP at IoU=0.5 per class, average across classes with at least one pseudo-GT.
    """
    frame_count = len(detections_per_frame)
    # Build pseudo ground truth per class per frame
    gt_by_class = {}  # class_id -> dict(frame_idx -> list of gt boxes)
    preds_by_class = {}  # class_id -> list of (frame_idx, score, box)

    for f_idx in range(frame_count):
        dets = detections_per_frame[f_idx]
        # Group by class
        cls_to_boxes = {}
        cls_to_scores = {}
        for d in dets:
            cid = d['class_id']
            cls_to_boxes.setdefault(cid, []).append(d['box'])
            cls_to_scores.setdefault(cid, []).append(d['score'])

        for cid, boxes in cls_to_boxes.items():
            scores = cls_to_scores[cid]
            # Pseudo-GT: keep boxes with score >= threshold, then NMS
            indices = [i for i, s in enumerate(scores) if s >= conf_threshold]
            sel_boxes = [boxes[i] for i in indices]
            sel_scores = [scores[i] for i in indices]
            keep = nms(sel_boxes, sel_scores, iou_threshold=iou_thresh)
            gt_boxes = [sel_boxes[i] for i in keep]
            if gt_boxes:
                gt_by_class.setdefault(cid, {}).setdefault(f_idx, []).extend(gt_boxes)

        # Predictions: all detections (without threshold) go into prediction pool
        for d in dets:
            cid = d['class_id']
            preds_by_class.setdefault(cid, []).append((f_idx, float(d['score']), d['box']))

    # Compute AP per class
    ap_list = []
    for cid in sorted(set(list(gt_by_class.keys()) + list(preds_by_class.keys()))):
        gt_frames = gt_by_class.get(cid, {})
        preds = preds_by_class.get(cid, [])

        # Count total GT
        total_gt = sum(len(v) for v in gt_frames.values())
        if total_gt == 0:
            continue  # skip classes without pseudo-GT

        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

        # Assigned GT flags
        assigned = {}  # (frame_idx, gt_idx) -> bool
        tp = []
        fp = []

        for (f_idx, score, pbox) in preds_sorted:
            gts = gt_frames.get(f_idx, [])
            best_iou = 0.0
            best_idx = -1
            for gi, gbox in enumerate(gts):
                current_iou = iou(pbox, gbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = gi
            if best_iou >= iou_thresh and best_idx >= 0 and not assigned.get((f_idx, best_idx), False):
                tp.append(1)
                fp.append(0)
                assigned[(f_idx, best_idx)] = True
            else:
                tp.append(0)
                fp.append(1)

        if len(tp) == 0:
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / float(total_gt)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

        # VOC 2007 11-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            prec_at_recall = precisions[recalls >= r]
            p = np.max(prec_at_recall) if prec_at_recall.size > 0 else 0.0
            ap += p
        ap /= 11.0
        ap_list.append(ap)

    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(ap_list))

def color_for_class(cid):
    np.random.seed(cid + 12345)
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return color

# =========================
# Main Application
# =========================
def main():
    print("TFLite object detection with TPU")
    print("Input description: Read a single video file from the given input_path")
    print("Output description: Output the video file with rectangles and labels, along with calculated mAP (mean average precision)")
    print(f"Model: {model_path}")
    print(f"Labels: {label_path}")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Confidence threshold: {confidence_threshold}")

    # Load EdgeTPU delegate and create interpreter
    delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    if not os.path.exists(delegate_path):
        raise FileNotFoundError(f"EdgeTPU delegate not found at {delegate_path}")
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(delegate_path)]
    )
    interpreter.allocate_tensors()

    # Model input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = int(input_details[0]['shape'][1])
    input_width = int(input_details[0]['shape'][2])
    input_dtype = input_details[0]['dtype']

    # Load labels
    labels = load_labels(label_path)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    detections_per_frame = []

    frame_index = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Keep a copy of original frame
        frames.append(frame.copy())

        # Preprocess
        input_data = preprocess_frame(frame, (input_height, input_width), input_dtype)

        # Inference
        set_input_tensor(interpreter, input_data)
        interpreter.invoke()

        # Postprocess
        boxes, classes, scores, num = get_output_tensors(interpreter)
        dets = []
        if boxes is not None and scores is not None and classes is not None:
            # Iterate all available detections (num may be <= len(scores))
            N = min(num, len(scores))
            img_h, img_w = frame.shape[0], frame.shape[1]
            for i in range(N):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                cls_id = int(classes[i])
                box = boxes[i]  # normalized ymin, xmin, ymax, xmax
                abs_box = denormalize_box(box, img_w, img_h)
                dets.append({'box': abs_box, 'score': score, 'class_id': cls_id})
        detections_per_frame.append(dets)

        frame_index += 1
        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames...")

    cap.release()
    elapsed = time.time() - t0
    print(f"Finished inference on {frame_index} frames in {elapsed:.2f}s ({(frame_index/elapsed if elapsed>0 else 0):.2f} FPS)")

    # Compute mAP using pseudo ground truth constructed from detections (IoU=0.5)
    print("Computing mAP (pseudo ground truth from detections)...")
    map_value = compute_map_pseudogt(detections_per_frame, (width, height), conf_threshold=confidence_threshold, iou_thresh=0.5)
    print(f"mAP@0.5 (pseudo-GT): {map_value:.4f}")

    # Prepare VideoWriter
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25.0, (width, height))

    # Draw and write frames with detections and mAP overlay
    for idx, frame in enumerate(frames):
        dets = detections_per_frame[idx]
        # Draw detections
        for d in dets:
            x1, y1, x2, y2 = d['box']
            score = d['score']
            cid = d['class_id']
            color = color_for_class(cid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = labels.get(cid, f"id:{cid}")
            text = f"{label_text}: {score:.2f}"
            # Text background
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, thickness=-1)
            cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # mAP overlay (constant final value)
        map_text = f"mAP@0.5 (pseudo-GT): {map_value:.3f}"
        cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 180, 20), 2, cv2.LINE_AA)

        writer.write(frame)

        if (idx + 1) % 100 == 0:
            print(f"Wrote {idx + 1}/{len(frames)} frames to output...")

    writer.release()
    print(f"Output saved to: {output_path}")
    print("Done.")

if __name__ == "__main__":
    main()