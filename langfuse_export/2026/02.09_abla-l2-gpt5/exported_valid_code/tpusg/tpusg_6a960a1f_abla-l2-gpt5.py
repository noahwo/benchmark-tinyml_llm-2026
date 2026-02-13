import os
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# --------------------------
# Configuration Parameters
# --------------------------
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# Optional: ground truth CSV to compute mAP if available.
# Expected CSV format per line: frame_index,xmin,ymin,xmax,ymax,class_id
# Coordinates are pixel values in the original frame size.
GT_PATH = os.getenv("GT_PATH", os.path.splitext(INPUT_PATH)[0] + "_gt.csv")


# --------------------------
# Utilities
# --------------------------
def load_labels(label_path):
    labels = {}
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Try formats:
            # 1) "id label"
            # 2) "id: label"
            # 3) "label" (implied id = line index)
            if ":" in line:
                parts = line.split(":", 1)
                try:
                    idx = int(parts[0].strip())
                    labels[idx] = parts[1].strip()
                except ValueError:
                    labels[i] = line
            else:
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                else:
                    labels[i] = line
    return labels


def make_interpreter(model_path, edgetpu_lib):
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(edgetpu_lib)]
    )
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame_bgr, input_details):
    h_in, w_in = input_details['shape'][1], input_details['shape'][2]
    dtype = input_details['dtype']
    # Convert BGR to RGB as most TFLite models expect RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (w_in, h_in))
    if dtype == np.float32:
        input_tensor = resized.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        input_tensor = resized.astype(np.uint8)
    else:
        # Fallback: convert to required dtype without scaling
        input_tensor = resized.astype(dtype)
    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def extract_detections(interpreter, frame_w, frame_h, confidence_threshold):
    # Typical EdgeTPU SSD outputs: boxes, classes, scores, num_detections
    output_details = interpreter.get_output_details()

    # Assume conventional ordering (EdgeTPU models generally follow this):
    # 0: boxes [1, num, 4], 1: classes [1, num], 2: scores [1, num], 3: num [1]
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num = int(interpreter.get_tensor(output_details[3]['index'])[0])

    detections = []
    for i in range(num):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        cls_id = int(classes[i])
        # Boxes are [ymin, xmin, ymax, xmax] in normalized coordinates
        ymin, xmin, ymax, xmax = boxes[i]
        # Convert to pixel coordinates in original frame size
        x1 = int(max(0, xmin * frame_w))
        y1 = int(max(0, ymin * frame_h))
        x2 = int(min(frame_w - 1, xmax * frame_w))
        y2 = int(min(frame_h - 1, ymax * frame_h))
        # Ensure valid box
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": cls_id
        })
    return detections


def draw_detections(frame, detections, labels):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cls_id = det["class_id"]
        label = labels.get(cls_id, f"id:{cls_id}")
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        t_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_br = (x1 + t_size[0] + 4, y1 - t_size[1] - 6)
        text_bg_tl = (x1, y1 - t_size[1] - 6 if y1 - t_size[1] - 6 > 0 else 0)
        # Draw filled background for readability
        cv2.rectangle(frame, text_bg_tl, text_bg_br, color, thickness=-1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4 if y1 - 4 > 0 else y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def load_ground_truth(gt_path):
    # Returns dict: frame_index -> list of dicts: {"bbox":(x1,y1,x2,y2), "class_id":int}
    gts = {}
    if not os.path.exists(gt_path):
        return None
    try:
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                fi = int(parts[0])
                x1, y1, x2, y2 = map(int, parts[1:5])
                cid = int(parts[5])
                gts.setdefault(fi, []).append({"bbox": (x1, y1, x2, y2), "class_id": cid})
        return gts
    except Exception:
        # If parsing fails, return None to skip mAP
        return None


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    boxA_area = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxB_area = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = float(boxA_area + boxB_area - inter_area + 1e-9)
    return inter_area / denom


def compute_map(predictions, ground_truths, iou_thresh=0.5):
    """
    predictions: list of dicts with keys: frame_index, class_id, score, bbox
    ground_truths: dict frame_index -> list of dicts with keys: class_id, bbox
    Returns mAP across classes that have GTs using 11-point interpolation.
    """
    if not ground_truths:
        return None

    # Organize GTs by class and frame
    gt_by_class_frame = {}
    total_gt_per_class = {}
    for fi, anns in ground_truths.items():
        for ann in anns:
            cid = ann["class_id"]
            gt_by_class_frame.setdefault(cid, {}).setdefault(fi, [])
            gt_by_class_frame[cid][fi].append({"bbox": ann["bbox"], "matched": False})
            total_gt_per_class[cid] = total_gt_per_class.get(cid, 0) + 1

    # Organize predictions by class
    preds_by_class = {}
    for p in predictions:
        cid = p["class_id"]
        preds_by_class.setdefault(cid, []).append(p)

    aps = []
    for cid, preds in preds_by_class.items():
        if cid not in total_gt_per_class or total_gt_per_class[cid] == 0:
            # No GT for this class -> skip from mAP
            continue

        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: -x["score"])
        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        # Reset matched flags for this class before evaluation
        for fi in gt_by_class_frame.get(cid, {}):
            for entry in gt_by_class_frame[cid][fi]:
                entry["matched"] = False

        for i, pred in enumerate(preds_sorted):
            fi = pred["frame_index"]
            pred_box = pred["bbox"]
            matched = False
            if cid in gt_by_class_frame and fi in gt_by_class_frame[cid]:
                candidates = gt_by_class_frame[cid][fi]
                best_iou = 0.0
                best_idx = -1
                for j, g in enumerate(candidates):
                    if g["matched"]:
                        continue
                    iou_val = iou(pred_box, g["bbox"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = j
                if best_iou >= iou_thresh and best_idx >= 0:
                    candidates[best_idx]["matched"] = True
                    matched = True
            if matched:
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        total_gt = float(total_gt_per_class[cid])
        recall = cum_tp / (total_gt + 1e-9)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)

        # 11-point interpolation AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recall >= t):
                p = np.max(precision[recall >= t])
            else:
                p = 0.0
            ap += p
        ap /= 11.0
        aps.append(ap)

    if not aps:
        return None
    return float(np.mean(aps))


# --------------------------
# Main Pipeline
# --------------------------
def main():
    # Setup: interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_LIB)
    input_details = interpreter.get_input_details()[0]

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {INPUT_PATH}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Load optional ground-truth
    ground_truths = load_ground_truth(GT_PATH)

    # Storage for predictions to compute mAP over time
    all_predictions = []

    frame_index = 0
    t0_total = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocessing
            input_tensor = preprocess_frame(frame, input_details)

            # Inference
            interpreter.set_tensor(input_details['index'], input_tensor)
            t0 = time.time()
            interpreter.invoke()
            infer_time_ms = (time.time() - t0) * 1000.0

            # Postprocessing: extract detections
            detections = extract_detections(interpreter, frame_w=orig_w, frame_h=orig_h,
                                            confidence_threshold=CONFIDENCE_THRESHOLD)

            # Accumulate predictions for mAP
            for det in detections:
                all_predictions.append({
                    "frame_index": frame_index,
                    "class_id": det["class_id"],
                    "score": det["score"],
                    "bbox": det["bbox"]
                })

            # Compute mAP if GT is available
            map_value = None
            if ground_truths is not None:
                # Use all frames up to current for progressive mAP
                # Filter GTs up to current frame
                partial_gt = {fi: anns for fi, anns in ground_truths.items() if fi <= frame_index}
                map_value = compute_map(all_predictions, partial_gt, iou_thresh=0.5)

            # Draw outputs
            draw_detections(frame, detections, labels)

            # Overlay performance and mAP
            info_lines = []
            info_lines.append(f"Infer: {infer_time_ms:.1f} ms")
            if map_value is None:
                info_lines.append("mAP: N/A (no ground truth)")
            else:
                info_lines.append(f"mAP@0.5: {map_value:.3f}")

            y0 = 24
            for line in info_lines:
                cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                y0 += 22

            # Write to output
            writer.write(frame)

            frame_index += 1

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0_total
    print(f"Processing completed. Frames: {frame_index}, Time: {total_time:.2f}s, FPS (overall): {frame_index / max(total_time, 1e-6):.2f}")
    print(f"Saved output video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()