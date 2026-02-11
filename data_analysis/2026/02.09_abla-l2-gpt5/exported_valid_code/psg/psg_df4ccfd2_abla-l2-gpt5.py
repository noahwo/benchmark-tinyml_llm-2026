import os
import time
import json
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# Utility Functions
# =========================
def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def load_labels(label_file):
    labels = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def get_label_name(labels, class_id):
    if class_id is None:
        return "N/A"
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return str(class_id)

def iou_yxyx(box_a, box_b):
    # boxes in [ymin, xmin, ymax, xmax], normalized or absolute equally applied
    ya1, xa1, ya2, xa2 = box_a
    yb1, xb1, yb2, xb2 = box_b

    inter_y1 = max(ya1, yb1)
    inter_x1 = max(xa1, xb1)
    inter_y2 = min(ya2, yb2)
    inter_x2 = min(xa2, xb2)

    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_area = inter_h * inter_w

    area_a = max(0.0, (ya2 - ya1)) * max(0.0, (xa2 - xa1))
    area_b = max(0.0, (yb2 - yb1)) * max(0.0, (xb2 - xb1))

    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom

def compute_ap(rec, prec):
    # VOC-style AP (area under precision-recall curve with interpolation)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Integrate area
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def compute_map(preds, gts_by_frame, iou_thresh=0.5, max_frame=None):
    """
    preds: list of dicts: {image_id:int, class_id:int, score:float, bbox:[ymin,xmin,ymax,xmax] normalized}
    gts_by_frame: dict[int] -> dict[class_id] -> list of gt boxes [ymin,xmin,ymax,xmax] normalized
    max_frame: consider frames up to and including this index; if None, consider all
    """
    # Filter by frame if needed
    if max_frame is not None:
        preds_filtered = [p for p in preds if p["image_id"] <= max_frame]
        considered_frames = {fid: cls_map for fid, cls_map in gts_by_frame.items() if fid <= max_frame}
    else:
        preds_filtered = preds
        considered_frames = gts_by_frame

    # Build GT dict: key=(image_id, class_id) -> {'boxes': [...], 'matched': [False]*N}
    gt_dict = {}
    gt_count_per_class = {}
    for fid, cls_map in considered_frames.items():
        for cid, boxes in cls_map.items():
            key = (fid, cid)
            gt_dict[key] = {"boxes": list(boxes), "matched": [False] * len(boxes)}
            gt_count_per_class[cid] = gt_count_per_class.get(cid, 0) + len(boxes)

    # Collect classes present in preds or GTs
    classes = set([p["class_id"] for p in preds_filtered]) | set(gt_count_per_class.keys())

    ap_by_class = {}
    valid_class_aps = []

    for cid in sorted(classes):
        # Predictions of this class
        preds_c = [p for p in preds_filtered if p["class_id"] == cid]
        if not preds_c and gt_count_per_class.get(cid, 0) == 0:
            continue  # nothing to evaluate

        # Sort predictions by descending score
        preds_c.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(preds_c), dtype=np.float32)
        fp = np.zeros(len(preds_c), dtype=np.float32)

        for i, p in enumerate(preds_c):
            key = (p["image_id"], cid)
            gt_entry = gt_dict.get(key, None)
            if gt_entry is None or len(gt_entry["boxes"]) == 0:
                # No GT for this class in this frame -> FP
                fp[i] = 1.0
                continue

            # Find best match IoU among unmatched GTs
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gt_entry["boxes"]):
                if gt_entry["matched"][j]:
                    continue
                iou = iou_yxyx(p["bbox"], g)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thresh and best_j >= 0:
                gt_entry["matched"][best_j] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        npos = gt_count_per_class.get(cid, 0)
        if npos == 0:
            # No GT for this class in considered frames -> AP undefined; skip from mAP
            ap_by_class[cid] = None
            continue

        # Precision-recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / float(npos)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        ap = compute_ap(rec, prec)
        ap_by_class[cid] = ap
        valid_class_aps.append(ap)

    if len(valid_class_aps) == 0:
        mean_ap = None
    else:
        mean_ap = float(np.mean(valid_class_aps))

    return mean_ap, ap_by_class

def random_color_for_id(idx):
    # Deterministic pseudo-random color from class id
    np.random.seed((idx * 123457) % 2**32)
    c = np.random.randint(0, 255, size=3).tolist()
    return (int(c[2]), int(c[1]), int(c[0]))  # BGR for OpenCV

def parse_optional_ground_truth(gt_path, frame_w, frame_h):
    """
    Optional ground truth JSON format:
    {
      "frames": {
        "0": [{"bbox": [ymin, xmin, ymax, xmax], "class_id": int, "normalized": true}, ...],
        "1": [...]
      }
    }
    - If "normalized" field missing, bbox is assumed normalized in [0,1].
    - If "normalized" is false, bbox is in absolute pixels (x/y in pixels), will be converted to normalized.
    """
    if not os.path.exists(gt_path):
        return {}, False

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames_obj = data.get("frames", {})
    gts_by_frame = {}
    for k, items in frames_obj.items():
        try:
            fid = int(k)
        except Exception:
            continue
        cls_map = {}
        for obj in items:
            bbox = obj.get("bbox", None)
            cid = obj.get("class_id", None)
            if bbox is None or cid is None:
                continue
            normalized_flag = obj.get("normalized", True)
            # Support [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax] if user accidentally uses xyxy
            # We'll assume provided is [ymin, xmin, ymax, xmax]; no robust inference beyond that.
            y1, x1, y2, x2 = bbox
            if not normalized_flag:
                # Convert absolute pixel to normalized based on provided frame size
                y1 = float(y1) / float(frame_h)
                y2 = float(y2) / float(frame_h)
                x1 = float(x1) / float(frame_w)
                x2 = float(x2) / float(frame_w)
            # Clamp to [0,1]
            y1 = max(0.0, min(1.0, float(y1)))
            y2 = max(0.0, min(1.0, float(y2)))
            x1 = max(0.0, min(1.0, float(x1)))
            x2 = max(0.0, min(1.0, float(x2)))
            if y2 <= y1 or x2 <= x1:
                continue
            cls_map.setdefault(int(cid), []).append([y1, x1, y2, x2])
        gts_by_frame[fid] = cls_map
    return gts_by_frame, True

# =========================
# Main Pipeline
# =========================
def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    input_index = input_details["index"]
    in_shape = input_details["shape"]
    # Expect [1, height, width, 3]
    in_height = int(in_shape[1])
    in_width = int(in_shape[2])
    in_dtype = input_details["dtype"]

    # Identify output indices by shape semantics
    out_indices = {"boxes": None, "classes": None, "scores": None, "num": None}
    for od in output_details:
        shp = od["shape"]
        if len(shp) == 3 and shp[-1] == 4:
            out_indices["boxes"] = od["index"]
        elif len(shp) == 2:
            # Could be classes or scores: need dtype heuristic
            # Classes often float32; scores float32 too. Distinguish by name if available.
            name = od.get("name", "").lower()
            if "class" in name:
                out_indices["classes"] = od["index"]
            elif "score" in name:
                out_indices["scores"] = od["index"]
            else:
                # Fallback by checking quantization/shape later
                # We'll assign the one not yet set to scores, the other to classes
                if out_indices["scores"] is None:
                    out_indices["scores"] = od["index"]
                else:
                    out_indices["classes"] = od["index"]
        elif len(shp) == 1 and shp[0] == 1:
            out_indices["num"] = od["index"]

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2 or np.isnan(fps):
        fps = 25.0  # Fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ensure_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    # Try loading optional ground truth (derived path: input without extension + '_gt.json')
    base_no_ext, _ = os.path.splitext(input_path)
    gt_path_guess = base_no_ext + "_gt.json"
    gts_by_frame, has_gt = parse_optional_ground_truth(gt_path_guess, frame_w, frame_h)

    # Accumulators for detections
    preds_all = []  # list of dicts: {image_id, class_id, score, bbox=[ymin,xmin,ymax,xmax] normalized}

    # Visualization parameters
    thickness = max(1, int(round(0.002 * (frame_w + frame_h) / 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.0, frame_h / 720.0))
    text_thickness = max(1, int(round(thickness)))

    print("Starting inference...")
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (in_width, in_height), interpolation=cv2.INTER_LINEAR)
        input_tensor = np.expand_dims(resized, axis=0)
        if in_dtype == np.float32:
            input_tensor = (input_tensor.astype(np.float32) / 255.0).astype(np.float32)
        else:
            input_tensor = input_tensor.astype(in_dtype)

        # Inference
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

        # Fetch outputs
        boxes = interpreter.get_tensor(out_indices["boxes"])  # [1, num, 4]
        classes = interpreter.get_tensor(out_indices["classes"])  # [1, num]
        scores = interpreter.get_tensor(out_indices["scores"])  # [1, num]
        num = interpreter.get_tensor(out_indices["num"])  # [1]

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        det_count = int(np.squeeze(num))

        # Draw detections and accumulate predictions
        for i in range(det_count):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            cls_id = int(classes[i])
            y1, x1, y2, x2 = boxes[i].tolist()

            # Clip and convert to pixel coords for drawing
            y1c = int(max(0, min(1, y1)) * frame_h)
            y2c = int(max(0, min(1, y2)) * frame_h)
            x1c = int(max(0, min(1, x1)) * frame_w)
            x2c = int(max(0, min(1, x2)) * frame_w)

            color = random_color_for_id(cls_id)
            cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, thickness)

            label_text = f"{get_label_name(labels, cls_id)}: {score:.2f}"
            # Text background
            (tw, th), bl = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            y_text = max(th + 2, y1c + th + 2)
            x_text = x1c
            cv2.rectangle(frame, (x_text, y_text - th - 4), (x_text + tw + 4, y_text + 2), color, -1)
            cv2.putText(frame, label_text, (x_text + 2, y_text - 2), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

            # Accumulate prediction in normalized coords for evaluation
            preds_all.append({
                "image_id": frame_idx,
                "class_id": cls_id,
                "score": score,
                "bbox": [max(0.0, min(1.0, y1)),
                         max(0.0, min(1.0, x1)),
                         max(0.0, min(1.0, y2)),
                         max(0.0, min(1.0, x2))]
            })

        # Compute mAP up to current frame if GT available
        map_text = "mAP@0.5: N/A"
        if has_gt:
            mean_ap, _ = compute_map(preds_all, gts_by_frame, iou_thresh=0.5, max_frame=frame_idx)
            if mean_ap is not None:
                map_text = f"mAP@0.5: {mean_ap:.3f}"
            else:
                map_text = "mAP@0.5: N/A"

        # Draw mAP on frame
        (tw, th), bl = cv2.getTextSize(map_text, font, font_scale * 1.1, text_thickness + 1)
        cv2.rectangle(frame, (8, 8), (8 + tw + 8, 8 + th + 12), (0, 0, 0), -1)
        cv2.putText(frame, map_text, (12, 8 + th + 2), font, font_scale * 1.1, (0, 255, 0), text_thickness + 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    elapsed = time.time() - start_time
    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames in {elapsed:.2f}s ({(frame_idx/elapsed) if elapsed>0 else 0:.2f} FPS).")
    if has_gt:
        final_map, _ = compute_map(preds_all, gts_by_frame, iou_thresh=0.5, max_frame=frame_idx - 1)
        if final_map is not None:
            print(f"Final mAP@0.5 over evaluated frames: {final_map:.4f}")
        else:
            print("Final mAP@0.5: N/A (no ground truth boxes found).")
    else:
        print("No ground truth file found; mAP overlays set to N/A.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()