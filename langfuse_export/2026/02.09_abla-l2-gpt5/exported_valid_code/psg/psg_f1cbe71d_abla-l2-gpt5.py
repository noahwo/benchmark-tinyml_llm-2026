import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
output_path = "results/object_detection/test_results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP text
confidence_threshold = 0.5

# =========================
# Utility Functions
# =========================
def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle "index label" or just "label"
            parts = line.split()
            if len(parts) > 1 and parts[0].isdigit():
                labels.append(" ".join(parts[1:]))
            else:
                labels.append(line)
    return labels

def preprocess_frame(frame_bgr, input_shape, input_dtype):
    # Model expects RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _, in_h, in_w, _ = input_shape
    resized = cv2.resize(rgb, (in_w, in_h))
    if input_dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    else:
        inp = resized.astype(input_dtype)
    return np.expand_dims(inp, axis=0)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def nms_per_class(boxes, scores, iou_threshold=0.5):
    # boxes: (N, 4) in pixel coords (ymin, xmin, ymax, xmax)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        inter = h * w
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def iou_matrix(boxes_a, boxes_b):
    # boxes: (N, 4) and (M, 4) in pixel coords (ymin, xmin, ymax, xmax)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    ay1, ax1, ay2, ax2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    by1, bx1, by2, bx2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_y1 = np.maximum(ay1, by1)
    inter_x1 = np.maximum(ax1, bx1)
    inter_y2 = np.minimum(ay2, by2)
    inter_x2 = np.minimum(ax2, bx2)

    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    inter = inter_h * inter_w

    area_a = (ay2 - ay1 + 1) * (ax2 - ax1 + 1)
    area_b = (by2 - by1 + 1) * (bx2 - bx1 + 1)

    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)

def parse_optional_gt_file(video_path):
    # Optional ground truth file alongside the video: replace extension with .gt.txt
    # Format per line (CSV or whitespace): frame_index, class_id, xmin, ymin, xmax, ymax
    base, _ = os.path.splitext(video_path)
    gt_path = base + ".gt.txt"
    if not os.path.exists(gt_path):
        return None
    gts = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) < 6:
                continue
            try:
                frame_idx = int(parts[0])
                cls = int(parts[1])
                xmin = int(float(parts[2]))
                ymin = int(float(parts[3]))
                xmax = int(float(parts[4]))
                ymax = int(float(parts[5]))
            except Exception:
                continue
            if frame_idx not in gts:
                gts[frame_idx] = []
            gts[frame_idx].append((cls, [ymin, xmin, ymax, xmax]))  # store as (class, [ymin,xmin,ymax,xmax])
    return gts

class MAPAccumulator:
    def __init__(self):
        # class_id -> dict with keys: scores (list), tps (list), total_gt (int)
        self.data = {}

    def ensure_class(self, cls_id):
        if cls_id not in self.data:
            self.data[cls_id] = {'scores': [], 'tps': [], 'total_gt': 0}

    def add_ground_truths(self, gt_by_class):
        # gt_by_class: dict class_id -> list of boxes
        for cls, boxes in gt_by_class.items():
            self.ensure_class(cls)
            self.data[cls]['total_gt'] += len(boxes)

    def add_detections(self, det_by_class, gt_by_class, iou_thresh=0.5):
        # det_by_class: dict class_id -> list of (score, box)
        # gt_by_class: dict class_id -> list of boxes
        for cls, dets in det_by_class.items():
            self.ensure_class(cls)
            gts = gt_by_class.get(cls, [])
            matched = np.zeros(len(gts), dtype=np.uint8) if gts else None
            # sort detections by score desc for consistent matching
            if dets:
                scores = np.array([d[0] for d in dets], dtype=np.float32)
                order = scores.argsort()[::-1]
                dets_sorted = [dets[i] for i in order]
            else:
                dets_sorted = []

            for score, box in dets_sorted:
                self.data[cls]['scores'].append(float(score))
                if not gts:
                    self.data[cls]['tps'].append(0)
                    continue
                # match to best unmatched gt by IoU
                ious = iou_matrix([box], gts)[0]
                # Invalidate already matched
                if matched is not None:
                    ious = ious * (1 - matched)
                best_idx = int(np.argmax(ious)) if ious.size > 0 else -1
                best_iou = ious[best_idx] if ious.size > 0 else 0.0
                if best_iou >= iou_thresh and matched[best_idx] == 0:
                    self.data[cls]['tps'].append(1)
                    matched[best_idx] = 1
                else:
                    self.data[cls]['tps'].append(0)

    def compute_map(self):
        # Compute AP per class and then mean across classes with at least one GT
        aps = []
        for cls, rec in self.data.items():
            total_gt = rec['total_gt']
            if total_gt <= 0:
                continue  # skip classes without GT
            scores = np.array(rec['scores'], dtype=np.float32)
            tps = np.array(rec['tps'], dtype=np.int32)
            if scores.size == 0:
                aps.append(0.0)
                continue
            order = scores.argsort()[::-1]
            tps = tps[order]
            fps = 1 - tps

            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            recall = cum_tp / max(total_gt, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

            # Precision envelope
            # Make precision monotonically non-increasing from right to left
            for i in range(precision.size - 2, -1, -1):
                if precision[i] < precision[i + 1]:
                    precision[i] = precision[i + 1]

            # Compute AP as area under PR curve via step-wise integration
            # Insert (0,1) at start and (1,0) at end as in common practice
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0]))
            # Make envelope again for safety
            for i in range(mpre.size - 2, -1, -1):
                if mpre[i] < mpre[i + 1]:
                    mpre[i] = mpre[i + 1]
            # Sum over recall steps
            # Find points where recall changes
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
            aps.append(ap)
        if not aps:
            return None  # No GT available yet
        return float(np.mean(aps))

# =========================
# Main Pipeline
# =========================
def main():
    # Prepare output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Load optional ground truth annotations
    gt_all = parse_optional_gt_file(input_path)
    have_gt = gt_all is not None
    map_acc = MAPAccumulator() if have_gt else None

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open input video:", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print("Error: Cannot open output video for writing:", output_path)
        cap.release()
        return

    frame_index = 0
    running_map = None
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            inp = preprocess_frame(frame, input_shape, input_dtype)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()

            # Extract outputs robustly
            outs = [interpreter.get_tensor(d['index']) for d in output_details]

            boxes = None
            classes = None
            scores = None
            num = None
            # Identify outputs by shape heuristics
            for a in outs:
                if a.size == 1:
                    num = int(np.round(float(a.reshape(-1)[0])))
                elif a.ndim == 3 and a.shape[0] == 1 and a.shape[2] == 4:
                    boxes = a[0]
                elif a.ndim == 2 and a.shape[0] == 1:
                    # Defer deciding between classes and scores
                    pass

            # Find the two [1, N] tensors for classes and scores
            two_d = [a for a in outs if (a.ndim == 2 and a.shape[0] == 1 and a.size > 1)]
            if len(two_d) >= 2:
                a0 = two_d[0][0]
                a1 = two_d[1][0]
                max0 = float(np.max(a0)) if a0.size > 0 else 0.0
                max1 = float(np.max(a1)) if a1.size > 0 else 0.0
                # Scores are usually in [0,1], classes usually > 1
                if max0 <= 1.0 and max1 > 1.0:
                    scores = a0
                    classes = a1
                elif max1 <= 1.0 and max0 > 1.0:
                    scores = a1
                    classes = a0
                else:
                    # Fallback by dtype: classes often float but larger values; pick based on means
                    if float(np.mean(a0)) < float(np.mean(a1)):
                        scores, classes = a0, a1
                    else:
                        scores, classes = a1, a0
            elif len(two_d) == 1:
                # Some models return only boxes and scores without classes; fallback class id 0
                scores = two_d[0][0]
                classes = np.zeros_like(scores)

            if boxes is None or scores is None or classes is None:
                # Cannot parse outputs; skip drawing for this frame
                writer.write(frame)
                frame_index += 1
                continue

            if num is None or num <= 0 or num > boxes.shape[0]:
                num = boxes.shape[0]

            # Postprocess detections: filter by score and convert to pixel coords
            filtered = []
            for i in range(num):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                cls_id = int(classes[i])
                y_min, x_min, y_max, x_max = boxes[i]
                # boxes are normalized [0,1]; convert to pixel coords
                top = int(clamp(y_min, 0.0, 1.0) * frame_h)
                left = int(clamp(x_min, 0.0, 1.0) * frame_w)
                bottom = int(clamp(y_max, 0.0, 1.0) * frame_h)
                right = int(clamp(x_max, 0.0, 1.0) * frame_w)
                # Clamp and fix ordering if necessary
                t = clamp(min(top, bottom), 0, frame_h - 1)
                b = clamp(max(top, bottom), 0, frame_h - 1)
                l = clamp(min(left, right), 0, frame_w - 1)
                r = clamp(max(left, right), 0, frame_w - 1)
                filtered.append((cls_id, score, [t, l, b, r]))

            # Apply per-class NMS
            final_dets = []
            if filtered:
                # Group by class
                by_class = {}
                for cls_id, score, box in filtered:
                    by_class.setdefault(cls_id, []).append((score, box))
                for cls_id, dets in by_class.items():
                    if not dets:
                        continue
                    boxes_cls = [b for (_, b) in dets]
                    scores_cls = [s for (s, _) in dets]
                    keep_idx = nms_per_class(boxes_cls, scores_cls, iou_threshold=0.5)
                    for idx in keep_idx:
                        final_dets.append((cls_id, scores_cls[idx], boxes_cls[idx]))
            else:
                final_dets = []

            # Update running mAP if GT available
            if have_gt:
                gt_frame_entries = gt_all.get(frame_index, [])
                gt_by_class = {}
                for cls, box in gt_frame_entries:
                    gt_by_class.setdefault(cls, []).append(box)
                # Add GT counts first (for frames processed so far)
                map_acc.add_ground_truths(gt_by_class)
                # Convert detections to dict by class with (score, box)
                det_by_class = {}
                for cls_id, score, box in final_dets:
                    det_by_class.setdefault(cls_id, []).append((score, box))
                # Update matches
                map_acc.add_detections(det_by_class, gt_by_class, iou_thresh=0.5)
                running_map = map_acc.compute_map()

            # Draw detections
            for cls_id, score, box in final_dets:
                t, l, b, r = box
                color = (0, 255, 0)
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                # Resolve label text
                label_txt = None
                if 0 <= cls_id < len(labels):
                    label_txt = labels[cls_id]
                else:
                    # Some label files have a "background" at index 0; try offset by 1
                    if 0 <= cls_id + 1 < len(labels):
                        label_txt = labels[cls_id + 1]
                if label_txt is None:
                    label_txt = f"class_{cls_id}"
                caption = f"{label_txt}: {score:.2f}"
                # Put label
                y_text = max(0, t - 10)
                cv2.putText(frame, caption, (l, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Put mAP text
            if have_gt:
                if running_map is None:
                    map_text = "mAP: N/A (insufficient GT)"
                else:
                    map_text = f"mAP: {running_map:.3f}"
            else:
                map_text = "mAP: N/A (no ground truth)"
            cv2.putText(frame, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 220, 255), 2, cv2.LINE_AA)

            # Write frame
            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()
        elapsed = time.time() - t0
        if frame_index > 0 and elapsed > 0:
            print(f"Processed {frame_index} frames in {elapsed:.2f}s ({frame_index/elapsed:.2f} FPS).")
        if have_gt:
            final_map = map_acc.compute_map()
            if final_map is None:
                print("Final mAP: N/A (no ground truth annotations were found).")
            else:
                print(f"Final mAP over processed frames: {final_map:.4f}")
        print("Output saved to:", output_path)

if __name__ == "__main__":
    main()