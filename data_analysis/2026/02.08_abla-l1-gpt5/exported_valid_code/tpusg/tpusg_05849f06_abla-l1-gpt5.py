import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"  # Output with rectangles, labels, mAP
CONF_THRESHOLD = 0.5

EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


# =========================
# Utilities
# =========================
def load_labels(label_path):
    labels = {}
    try:
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Try formats like: "0 person" or "0: person"
                parts = line.replace(":", " ").split()
                if len(parts) >= 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = " ".join(parts[1:]).strip()
                    labels[idx] = name
                else:
                    labels[i] = line
    except Exception as e:
        print(f"Warning: Failed to load labels from {label_path}: {e}")
    return labels


def make_interpreter(model_path, delegate_path):
    # Try EdgeTPU delegate first; fallback to CPU if it fails.
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(delegate_path)]
        )
        print("Interpreter initialized with EdgeTPU delegate.")
        return interpreter
    except Exception as e:
        print(f"Warning: Failed to load EdgeTPU delegate ({delegate_path}): {e}")
        print("Falling back to CPU interpreter.")
        interpreter = Interpreter(model_path=model_path)
        return interpreter


def preprocess_frame(frame, input_size, input_dtype):
    # frame: BGR HxWx3
    # model expects RGB usually; resize to input_size
    h, w = input_size
    img = cv2.resize(frame, (w, h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        input_data = (img_rgb.astype(np.float32) / 255.0).reshape(1, h, w, 3)
    else:
        input_data = img_rgb.astype(np.uint8).reshape(1, h, w, 3)
    return input_data


def parse_detections(interpreter, frame_shape, conf_threshold=0.5):
    # Attempt to extract boxes, classes, scores, count from common TFLite detection outputs
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    # Initialize placeholders
    boxes = None
    classes = None
    scores = None
    count = None

    # Heuristic mapping based on typical SSD outputs
    for i, od in enumerate(output_details):
        tensor = outputs[i]
        shape = tensor.shape
        dtype = tensor.dtype

        # boxes: [1, N, 4] float
        if len(shape) == 3 and shape[-1] == 4:
            boxes = tensor[0]
        # classes: [1, N] float or int
        elif len(shape) == 2 and shape[0] == 1 and shape[1] >= 1:
            # Could be classes or scores; try dtype to separate
            if dtype == np.float32:
                # Could be scores or classes (as float). We'll detect scores separately.
                # We'll temporarily store, decide later.
                pass
        # scores: [1, N] float
        # count: [1] float/int
        if len(shape) == 2 and shape[0] == 1 and dtype == np.float32:
            # could be classes float (some models) or scores
            # Decide based on value ranges: scores within [0,1], classes often >=0 and fractional
            vals = tensor[0]
            if np.all((vals >= 0.0) & (vals <= 1.0)):
                scores = vals
            else:
                classes = vals.astype(np.int32)
        if len(shape) == 1 and shape[0] == 1:
            # count
            count = int(np.squeeze(tensor))

    # If classes is still None, try to locate int32 tensor
    if classes is None:
        for i, od in enumerate(output_details):
            tensor = outputs[i]
            if tensor.dtype in (np.int32, np.uint8) and len(tensor.shape) == 2 and tensor.shape[0] == 1:
                classes = tensor[0].astype(np.int32)
                break

    # If scores is still None, try to locate float array different from boxes
    if scores is None:
        for i, od in enumerate(output_details):
            tensor = outputs[i]
            if tensor.dtype == np.float32 and len(tensor.shape) == 2 and tensor.shape[0] == 1:
                if tensor.shape[1] != 4:  # not boxes
                    arr = tensor[0]
                    if np.all((arr >= 0.0) & (arr <= 1.0)):
                        scores = arr
                        break

    # If count is None, infer from boxes or scores
    if count is None:
        if boxes is not None:
            count = boxes.shape[0]
        elif scores is not None:
            count = scores.shape[0]
        else:
            count = 0

    detections = []
    H, W = frame_shape[:2]
    if boxes is None or scores is None or classes is None:
        return detections

    num = min(count, len(scores), len(classes), len(boxes))
    for i in range(num):
        score = float(scores[i])
        if score < conf_threshold:
            continue
        cls_id = int(classes[i])
        # boxes are [ymin, xmin, ymax, xmax] normalized [0,1]
        ymin, xmin, ymax, xmax = boxes[i]
        xmin = max(0, min(int(xmin * W), W - 1))
        xmax = max(0, min(int(xmax * W), W - 1))
        ymin = max(0, min(int(ymin * H), H - 1))
        ymax = max(0, min(int(ymax * H), H - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        detections.append({
            "class_id": cls_id,
            "score": score,
            "bbox": [xmin, ymin, xmax, ymax]
        })
    return detections


def color_for_class(c):
    # Deterministic color for class id
    np.random.seed((c + 3) * 997)
    col = np.random.randint(64, 255, size=3).tolist()
    return (int(col[0]), int(col[1]), int(col[2]))


def iou_xyxy(a, b):
    # a, b: [xmin, ymin, xmax, ymax]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area_a = max(0, (ax2 - ax1 + 1)) * max(0, (ay2 - ay1 + 1))
    area_b = max(0, (bx2 - bx1 + 1)) * max(0, (by2 - by1 + 1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_ap(rec, prec):
    # VOC-style AP computation with precision envelope
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def compute_map(ground_truth, predictions, iou_thresh=0.5):
    """
    ground_truth: dict[frame_idx] -> list of {class_id:int, bbox:[xmin,ymin,xmax,ymax]}
    predictions:  dict[frame_idx] -> list of {class_id:int, bbox:[...], score:float}
    returns: (mAP, per_class_AP) or (None, {}) if no ground truth available
    """
    # Build per-class structures
    gt_by_class = {}
    for fidx, gts in ground_truth.items():
        for gt in gts:
            cls = gt["class_id"]
            gt_by_class.setdefault(cls, {}).setdefault(fidx, []).append({"bbox": gt["bbox"], "matched": False})

    if len(gt_by_class) == 0:
        return None, {}

    pred_by_class = {}
    for fidx, preds in predictions.items():
        for pr in preds:
            cls = pr["class_id"]
            pred_by_class.setdefault(cls, []).append({"frame": fidx, "bbox": pr["bbox"], "score": pr["score"]})

    ap_per_class = {}
    for cls, gt_frames in gt_by_class.items():
        # Flatten gt count
        npos = sum(len(lst) for lst in gt_frames.values())
        if npos == 0:
            continue
        preds_cls = pred_by_class.get(cls, [])
        if len(preds_cls) == 0:
            ap_per_class[cls] = 0.0
            continue
        # Sort predictions by score desc
        preds_cls = sorted(preds_cls, key=lambda x: -x["score"])

        tp = np.zeros(len(preds_cls))
        fp = np.zeros(len(preds_cls))

        for i, p in enumerate(preds_cls):
            fidx = p["frame"]
            bbox_p = p["bbox"]
            gts = gt_frames.get(fidx, [])
            best_iou = 0.0
            best_gt_idx = -1
            for j, g in enumerate(gts):
                iou = iou_xyxy(bbox_p, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                if not gts[best_gt_idx]["matched"]:
                    tp[i] = 1.0
                    gts[best_gt_idx]["matched"] = True
                else:
                    fp[i] = 1.0  # duplicate detection
            else:
                fp[i] = 1.0
        # Precision-recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / float(npos)
        prec = np.divide(cum_tp, (cum_tp + cum_fp + 1e-10))
        ap = compute_ap(rec, prec)
        ap_per_class[cls] = ap

    if len(ap_per_class) == 0:
        return 0.0, {}
    mAP = float(np.mean(list(ap_per_class.values())))
    return mAP, ap_per_class


# =========================
# Main pipeline
# =========================
def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Setup interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_SHARED_LIB)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    if not input_details:
        print("Error: Interpreter has no input details.")
        return
    input_idx = input_details[0]['index']
    input_shape = input_details[0]['shape']  # [1, h, w, 3]
    input_dtype = input_details[0]['dtype']
    in_h, in_w = int(input_shape[1]), int(input_shape[2])

    # Open video input
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {INPUT_PATH}")
        return

    # Get input video properties
    in_w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w_frame, in_h_frame))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for write: {OUTPUT_PATH}")
        cap.release()
        return

    # Structures for mAP computation
    # Ground truth is not provided in configuration; keep empty.
    ground_truth_by_frame = {}  # dict[int] -> list of {"class_id":int, "bbox":[xmin,ymin,xmax,ymax]}
    predictions_by_frame = {}   # dict[int] -> list of {"class_id":int, "bbox":[...], "score":float}

    # Processing loop
    frame_index = 0
    inference_times = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_data = preprocess_frame(frame, (in_h, in_w), input_dtype)

        interpreter.set_tensor(input_idx, input_data)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        inference_times.append(t1 - t0)

        # Post-process: parse detections
        detections = parse_detections(interpreter, frame.shape, conf_threshold=CONF_THRESHOLD)

        # Store predictions for mAP
        predictions_by_frame[frame_index] = detections

        # Draw boxes and labels
        for det in detections:
            cls_id = det["class_id"]
            score = det["score"]
            xmin, ymin, xmax, ymax = det["bbox"]
            color = color_for_class(cls_id)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = labels.get(cls_id, f"id:{cls_id}")
            text = f"{label} {score:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, ymin), color, -1)
            cv2.putText(frame, text, (xmin + 2, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Compute running mAP if GT available (none provided -> will be None)
        mAP, _ = compute_map(ground_truth_by_frame, {k: predictions_by_frame[k] for k in range(frame_index + 1)}, iou_thresh=0.5)
        if mAP is None:
            map_text = "mAP: N/A (no GT)"
        else:
            map_text = f"mAP@0.5: {mAP:.3f}"
        cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)

        # Write frame
        writer.write(frame)
        frame_index += 1

    # Cleanup
    cap.release()
    writer.release()
    total_time = time.time() - start_time

    # Summary
    if inference_times:
        avg_inf_ms = (sum(inference_times) / len(inference_times)) * 1000.0
        print(f"Processed {frame_index} frames in {total_time:.2f}s. Avg inference: {avg_inf_ms:.2f} ms/frame")
    else:
        print("No frames processed.")

    # Final mAP over entire video
    final_mAP, _ = compute_map(ground_truth_by_frame, predictions_by_frame, iou_thresh=0.5)
    if final_mAP is None:
        print("Final mAP: N/A (no ground truth provided)")
    else:
        print(f"Final mAP@0.5 over video: {final_mAP:.4f}")


if __name__ == "__main__":
    main()