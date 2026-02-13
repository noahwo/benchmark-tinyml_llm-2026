import os
import time
import json
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration parameters
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5
EDGETPU_DELEGATE = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# =========================
# Utility functions
# =========================
def load_labels(path):
    labels = {}
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                else:
                    # Fallback: use line as label with incremental index
                    labels[i] = line
    except Exception as e:
        print(f"Warning: failed to load labels from {path}: {e}")
    return labels

def make_interpreter(model_path, delegate_path):
    delegates = []
    try:
        delegates.append(load_delegate(delegate_path))
    except Exception as e:
        print(f"Warning: Failed to load EdgeTPU delegate '{delegate_path}': {e}")
        print("Falling back to CPU. Inference will be slower.")
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates if delegates else None)
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    ih, iw = input_details["shape"][1], input_details["shape"][2]
    dtype = input_details["dtype"]
    return iw, ih, dtype

def preprocess_frame(frame_bgr, input_w, input_h, input_dtype):
    # Convert BGR to RGB (TFLite models typically trained on RGB) and resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h))
    if input_dtype == np.float32:
        # General normalization for float models
        input_tensor = (resized.astype(np.float32) - 127.5) / 127.5
    else:
        input_tensor = resized.astype(np.uint8)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def extract_detections(interpreter, frame_w, frame_h):
    # Attempt robust extraction of standard SSD outputs
    out_details = interpreter.get_output_details()
    idx_boxes = idx_scores = idx_classes = idx_count = None

    # Try match by name first
    for d in out_details:
        name = (d.get("name") or "").lower()
        if "box" in name:
            idx_boxes = d["index"]
        elif "score" in name:
            idx_scores = d["index"]
        elif "class" in name:
            idx_classes = d["index"]
        elif "count" in name or "num" in name:
            idx_count = d["index"]

    # Fallback by shapes if needed
    if idx_boxes is None or idx_scores is None or idx_classes is None:
        for d in out_details:
            shape = d["shape"]
            if len(shape) == 3 and shape[-1] == 4:
                idx_boxes = idx_boxes or d["index"]
            elif len(shape) == 2:
                # Heuristic: classes and scores are 2D [1, N]
                if idx_scores is None:
                    idx_scores = d["index"]
                elif idx_classes is None:
                    idx_classes = d["index"]
            elif len(shape) == 1:
                idx_count = idx_count or d["index"]

    boxes = interpreter.get_tensor(idx_boxes)[0] if idx_boxes is not None else np.zeros((0, 4), dtype=np.float32)
    classes = interpreter.get_tensor(idx_classes)[0] if idx_classes is not None else np.zeros((0,), dtype=np.float32)
    scores = interpreter.get_tensor(idx_scores)[0] if idx_scores is not None else np.zeros((0,), dtype=np.float32)
    num = int(interpreter.get_tensor(idx_count)[0]) if idx_count is not None else len(scores)

    detections = []
    for i in range(min(num, len(scores))):
        score = float(scores[i])
        if score < CONF_THRESHOLD:
            continue
        # boxes are y_min, x_min, y_max, x_max normalized
        y_min, x_min, y_max, x_max = boxes[i]
        x1 = int(max(0, min(frame_w - 1, x_min * frame_w)))
        y1 = int(max(0, min(frame_h - 1, y_min * frame_h)))
        x2 = int(max(0, min(frame_w - 1, x_max * frame_w)))
        y2 = int(max(0, min(frame_h - 1, y_max * frame_h)))
        class_id = int(classes[i])  # class indices are typically int
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "score": score
        })
    return detections

def draw_detections(frame, detections, labels, map_text=None, fps_text=None):
    # Simple color palette based on class_id
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        score = det["score"]
        color = tuple(int(c) for c in np.random.RandomState(class_id).randint(0, 255, size=3))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = labels.get(class_id, str(class_id))
        caption = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, caption, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay metrics (mAP and FPS) at top-left
    overlay_lines = []
    if map_text is not None:
        overlay_lines.append(f"mAP@0.5: {map_text}")
    if fps_text is not None:
        overlay_lines.append(f"FPS: {fps_text}")
    if overlay_lines:
        y = 20
        for line in overlay_lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y += 24
    return frame

def iou(box_a, box_b):
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
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def average_precision(recalls, precisions):
    # VOC-style AP computing: interpolation and integration
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Compute area under curve by summing delta recall * precision
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

def compute_map(predictions_by_frame, gt_by_frame, iou_threshold=0.5):
    # Flatten by class; treat each frame as separate image (standard for video -> image set)
    # Build ground truth structure per class and per frame
    gt_per_class = {}
    total_gt_per_class = {}
    for f_idx, gts in enumerate(gt_by_frame):
        for gt in gts:
            c = gt["class_id"]
            gt_per_class.setdefault(c, {})
            gt_per_class[c].setdefault(f_idx, [])
            gt_per_class[c][f_idx].append({"bbox": gt["bbox"], "matched": False})
            total_gt_per_class[c] = total_gt_per_class.get(c, 0) + 1

    ap_list = []
    classes_evaluated = 0

    # Build predictions per class
    preds_per_class = {}
    for f_idx, preds in enumerate(predictions_by_frame):
        for p in preds:
            c = p["class_id"]
            preds_per_class.setdefault(c, [])
            preds_per_class[c].append({"frame": f_idx, "bbox": p["bbox"], "score": p["score"]})

    for c, preds in preds_per_class.items():
        if c not in gt_per_class:
            # No ground truth for this class; skip from mAP calculation
            continue
        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
        tp = np.zeros(len(preds_sorted))
        fp = np.zeros(len(preds_sorted))
        gt_for_c = gt_per_class[c]
        total_gt = total_gt_per_class.get(c, 0)
        if total_gt == 0:
            continue

        for i, pred in enumerate(preds_sorted):
            fidx = pred["frame"]
            pbox = pred["bbox"]
            gts_in_frame = gt_for_c.get(fidx, [])
            iou_max = 0.0
            jmax = -1
            for j, gt in enumerate(gts_in_frame):
                iou_val = iou(pbox, gt["bbox"])
                if iou_val > iou_max:
                    iou_max = iou_val
                    jmax = j
            if iou_max >= iou_threshold and jmax >= 0 and not gts_in_frame[jmax]["matched"]:
                tp[i] = 1
                gts_in_frame[jmax]["matched"] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (total_gt + 1e-12)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        ap = average_precision(recalls, precisions)
        ap_list.append(ap)
        classes_evaluated += 1

    if classes_evaluated == 0:
        return None  # No ground truth available to compute mAP
    return float(np.mean(ap_list))

def try_load_ground_truth(input_video_path):
    # Attempt to find a sidecar JSON with ground-truth annotations
    # Expected simple format:
    # {
    #   "frames": [
    #       {"frame_index": 0, "objects":[{"bbox":[x1,y1,x2,y2], "class_id": int}, ...]},
    #       ...
    #   ]
    # }
    candidates = []
    base, _ = os.path.splitext(input_video_path)
    candidates.append(base + ".json")
    candidates.append(base + "_gt.json")
    candidates.append(base + "_groundtruth.json")
    candidates.append(base + "_annotations.json")

    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r") as f:
                    data = json.load(f)
                frames_data = data.get("frames", [])
                # Build gt_by_frame as a list indexed by frame order.
                # If some frames are missing, fill with empty list.
                if not frames_data:
                    continue
                max_index = max(fd.get("frame_index", 0) for fd in frames_data)
                gt_by_frame = [[] for _ in range(max_index + 1)]
                for fd in frames_data:
                    idx = int(fd.get("frame_index", 0))
                    objs = fd.get("objects", [])
                    clean_objs = []
                    for obj in objs:
                        bbox = obj.get("bbox", None)
                        cid = obj.get("class_id", None)
                        if bbox and cid is not None and len(bbox) == 4:
                            clean_objs.append({"bbox": [int(b) for b in bbox], "class_id": int(cid)})
                    gt_by_frame[idx] = clean_objs
                print(f"Loaded ground truth from: {cand}")
                return gt_by_frame
            except Exception as e:
                print(f"Warning: Failed to parse ground truth from {cand}: {e}")
                continue
    print("No ground-truth file found; mAP will be reported as N/A.")
    return None

# =========================
# Main application
# =========================
def main():
    # 1) Setup: interpreter, labels, video IO
    labels = load_labels(LABEL_PATH)
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_DELEGATE)
    input_w, input_h, input_dtype = get_input_details(interpreter)

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: could not open input video: {INPUT_PATH}")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0  # Fallback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: could not open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Optional: load ground truth for mAP
    gt_by_frame = try_load_ground_truth(INPUT_PATH)
    predictions_by_frame = []

    # Performance metrics
    t0 = time.time()
    fps_smooth = None
    frame_index = 0

    # 2,3,4) Process frames: preprocess, inference, draw, aggregate metrics, write out
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame, input_w, input_h, input_dtype)
        input_details = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_details["index"], input_tensor)
        t_start = time.time()
        interpreter.invoke()
        infer_time = time.time() - t_start

        detections = extract_detections(interpreter, frame_w, frame_h)
        predictions_by_frame.append(detections)

        # Compute running mAP if GT available; otherwise show N/A
        map_text = None
        if gt_by_frame is not None:
            # Ensure gt_by_frame has at least this frame index (pad empty if necessary)
            if frame_index >= len(gt_by_frame):
                gt_by_frame.extend([[] for _ in range(frame_index - len(gt_by_frame) + 1)])
            mAP_val = compute_map(predictions_by_frame, gt_by_frame[:len(predictions_by_frame)], iou_threshold=0.5)
            map_text = f"{mAP_val:.3f}" if mAP_val is not None else "N/A"
        else:
            map_text = "N/A"

        # Smooth FPS display
        inst_fps = 1.0 / max(infer_time, 1e-6)
        if fps_smooth is None:
            fps_smooth = inst_fps
        else:
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

        draw_detections(frame, detections, labels, map_text=map_text, fps_text=f"{fps_smooth:.1f}")
        writer.write(frame)

        frame_index += 1

    # Finalize and report
    cap.release()
    writer.release()

    # Final mAP computation for logging (printed)
    final_map = None
    if gt_by_frame is not None:
        # Pad ground truth if fewer frames than predictions
        if len(gt_by_frame) < len(predictions_by_frame):
            gt_by_frame.extend([[] for _ in range(len(predictions_by_frame) - len(gt_by_frame))])
        final_map = compute_map(predictions_by_frame, gt_by_frame, iou_threshold=0.5)

    elapsed = time.time() - t0
    print("Processing complete.")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_index}, time: {elapsed:.2f}s, avg FPS: {frame_index / max(elapsed,1e-6):.2f}")
    if final_map is None:
        print("mAP@0.5: N/A (no ground-truth found or invalid format)")
    else:
        print(f"mAP@0.5: {final_map:.4f}")

if __name__ == "__main__":
    main()