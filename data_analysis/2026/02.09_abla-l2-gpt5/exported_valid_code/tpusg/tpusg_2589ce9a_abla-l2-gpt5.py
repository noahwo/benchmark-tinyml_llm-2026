import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# ================== Configuration Parameters ==================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# Optional ground-truth for mAP (per-frame text annotations). If this file exists,
# it is used to compute mAP across frames.
# Format per line: frame_index class_id x_min y_min x_max y_max  (pixel coordinates)
GT_PATH = INPUT_PATH + ".gt.txt"

# ================== Utility Functions ==================
def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Try formats: "id label" or "label"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[i] = line
    return labels

def load_ground_truth(path):
    # Returns dict: frame_idx -> list of dicts {class_id, bbox:[x1,y1,x2,y2]}
    gts = {}
    if not os.path.exists(path):
        return gts
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 6:
                continue
            try:
                frame_idx = int(parts[0])
                class_id = int(parts[1])
                x1 = float(parts[2]); y1 = float(parts[3]); x2 = float(parts[4]); y2 = float(parts[5])
            except:
                continue
            if frame_idx not in gts:
                gts[frame_idx] = []
            gts[frame_idx].append({"class_id": class_id, "bbox": [x1, y1, x2, y2]})
    return gts

def make_interpreter(model_path, edgetpu_lib):
    return Interpreter(model_path=model_path, experimental_delegates=[load_delegate(edgetpu_lib)])

def set_input_tensor(interpreter, input_image_rgb):
    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    # Ensure dtype matches model
    if input_details["dtype"] == np.uint8:
        input_data = input_image_rgb.astype(np.uint8)
    else:
        # If model expects float, convert accordingly (rare for EdgeTPU)
        input_data = input_image_rgb.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_index, input_data)

def get_input_size(interpreter):
    input_details = interpreter.get_input_details()[0]
    # Expect shape: [1, height, width, channels]
    return int(input_details["shape"][2]), int(input_details["shape"][1])  # width, height

def detect_objects(interpreter, image_rgb, threshold):
    # Assumes typical EdgeTPU SSD detection outputs:
    # 0: boxes [1, num, 4], 1: classes [1, num], 2: scores [1, num], 3: count [1]
    set_input_tensor(interpreter, image_rgb)
    t0 = time.time()
    interpreter.invoke()
    infer_ms = (time.time() - t0) * 1000.0

    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    # Count may be float or int; ensure int
    raw_count = interpreter.get_tensor(output_details[3]["index"])
    count = int(raw_count.flatten()[0]) if raw_count.size > 0 else len(scores)

    results = []
    for i in range(count):
        score = float(scores[i])
        if score < threshold:
            continue
        cls = int(classes[i])
        ymin, xmin, ymax, xmax = boxes[i]
        results.append({
            "bbox_norm": [xmin, ymin, xmax, ymax],  # normalized [0..1]
            "score": score,
            "class_id": cls
        })
    return results, infer_ms

def norm_to_abs_bbox(bbox_norm, frame_w, frame_h):
    xmin_n, ymin_n, xmax_n, ymax_n = bbox_norm
    x1 = max(0, min(frame_w - 1, int(xmin_n * frame_w)))
    y1 = max(0, min(frame_h - 1, int(ymin_n * frame_h)))
    x2 = max(0, min(frame_w - 1, int(xmax_n * frame_w)))
    y2 = max(0, min(frame_h - 1, int(ymax_n * frame_h)))
    return [x1, y1, x2, y2]

def draw_prediction(frame, bbox, label_text, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    y_text = max(0, y1 - 5)
    cv2.rectangle(frame, (x1, y_text - text_h - baseline), (x1 + text_w + 2, y_text + 2), color, -1)
    cv2.putText(frame, label_text, (x1 + 1, y_text), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def compute_iou(boxA, boxB):
    # boxes: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = xB - xA + 1
    inter_h = yB - yA + 1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter_area = inter_w * inter_h
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = boxA_area + boxB_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def update_map_stats(stats, preds_frame, gts_frame, iou_thresh=0.5):
    # stats: class_id -> {"scores": [], "matches": [], "gt_count": int}
    # Greedy match per class within this frame
    # Prepare GT availability flags per class
    gt_by_class = {}
    for gt in gts_frame:
        cid = gt["class_id"]
        if cid not in gt_by_class:
            gt_by_class[cid] = []
        gt_by_class[cid].append({"bbox": gt["bbox"], "used": False})
    pred_by_class = {}
    for pr in preds_frame:
        cid = pr["class_id"]
        if cid not in pred_by_class:
            pred_by_class[cid] = []
        pred_by_class[cid].append(pr)

    # Update per-class stats
    for cid in set(list(gt_by_class.keys()) + list(pred_by_class.keys())):
        if cid not in stats:
            stats[cid] = {"scores": [], "matches": [], "gt_count": 0}
        gt_list = gt_by_class.get(cid, [])
        pr_list = pred_by_class.get(cid, [])

        # Count ground truths
        stats[cid]["gt_count"] += len(gt_list)

        # Sort predictions in this frame by score descending for greedy matching
        pr_list_sorted = sorted(pr_list, key=lambda x: x["score"], reverse=True)

        for pr in pr_list_sorted:
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gt_list):
                if gt["used"]:
                    continue
                iou = compute_iou(pr["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            is_tp = 0
            if best_iou >= iou_thresh and best_j >= 0:
                is_tp = 1
                gt_list[best_j]["used"] = True
            stats[cid]["scores"].append(float(pr["score"]))
            stats[cid]["matches"].append(int(is_tp))

def compute_ap(scores, matches, gt_count):
    # scores: list of floats
    # matches: list of 0/1 (1 means TP), ordered by any time, we will sort by scores desc
    if gt_count == 0:
        return None
    if len(scores) == 0:
        return 0.0
    scores_np = np.array(scores, dtype=np.float32)
    matches_np = np.array(matches, dtype=np.int32)

    order = np.argsort(-scores_np)
    matches_sorted = matches_np[order]

    tp_cum = np.cumsum(matches_sorted)
    fp_cum = np.cumsum(1 - matches_sorted)

    denom = tp_cum + fp_cum
    denom[denom == 0] = 1  # avoid div by zero
    precisions = tp_cum / denom
    recalls = tp_cum / float(gt_count)

    # VOC-style AP: integrate precision envelope over recall
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    # Precision envelope
    for i in range(mpre.size - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]
    # Compute area under PR curve
    ap = 0.0
    for i in range(1, mrec.size):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)

def compute_map(stats):
    # stats: class_id -> {"scores": [], "matches": [], "gt_count": int}
    aps = []
    for cid, d in stats.items():
        ap = compute_ap(d["scores"], d["matches"], d["gt_count"])
        if ap is not None:
            aps.append(ap)
    if len(aps) == 0:
        return None
    return float(np.mean(np.array(aps, dtype=np.float32)))

def pick_label_name(class_id, labels):
    # Try exact, then +1 offset
    if class_id in labels:
        return labels[class_id]
    if (class_id + 1) in labels:
        return labels[class_id + 1]
    return str(class_id)

# ================== Main Pipeline ==================
def main():
    # Prepare output directory
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Load optional ground-truth annotations
    gt_all = load_ground_truth(GT_PATH)
    has_gt = len(gt_all) > 0

    # Initialize TFLite Interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_LIB)
    interpreter.allocate_tensors()
    in_w, in_h = get_input_size(interpreter)

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open input video:", INPUT_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print("ERROR: Cannot open output video for writing:", OUTPUT_PATH)
        cap.release()
        return

    # mAP stats across frames
    map_stats = {}  # class_id -> {"scores": [], "matches": [], "gt_count": int}
    running_map = None

    frame_idx = 0
    last_map_str = "mAP: N/A"
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Preprocessing: resize to model input and convert BGR->RGB
            resized = cv2.resize(frame_bgr, (in_w, in_h))
            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Inference
            detections, infer_ms = detect_objects(interpreter, frame_rgb, CONFIDENCE_THRESHOLD)

            # Convert normalized boxes to absolute on original frame size
            preds_abs = []
            for det in detections:
                abs_box = norm_to_abs_bbox(det["bbox_norm"], frame_w, frame_h)
                preds_abs.append({
                    "bbox": abs_box,
                    "score": det["score"],
                    "class_id": det["class_id"]
                })

            # Prepare ground truths for this frame, if available
            gts_frame = gt_all.get(frame_idx, []) if has_gt else []

            # Update mAP stats and compute running mAP
            if has_gt:
                update_map_stats(map_stats, preds_abs, gts_frame, iou_thresh=0.5)
                running_map = compute_map(map_stats)
                if running_map is not None:
                    last_map_str = "mAP: {:.3f}".format(running_map)
                else:
                    last_map_str = "mAP: N/A"
            else:
                last_map_str = "mAP: N/A"

            # Draw detections
            overlay = frame_bgr.copy()
            for pr in preds_abs:
                cls_id = pr["class_id"]
                label = pick_label_name(cls_id, labels)
                conf = pr["score"]
                text = "{}: {:.2f}".format(label, conf)
                draw_prediction(overlay, pr["bbox"], text, color=(0, 255, 0))

            # Draw mAP and inference time
            info_text = "{} | {:.1f} ms".format(last_map_str, infer_ms)
            cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            # Write frame
            writer.write(overlay)

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # Print final summary
    if has_gt:
        final_map = compute_map(map_stats)
        if final_map is not None:
            print("Final mAP over processed video: {:.4f}".format(final_map))
        else:
            print("Final mAP: N/A (no valid ground-truth instances)")
    else:
        print("No ground-truth file found at {}. mAP not computed.".format(GT_PATH))
    print("Output saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()