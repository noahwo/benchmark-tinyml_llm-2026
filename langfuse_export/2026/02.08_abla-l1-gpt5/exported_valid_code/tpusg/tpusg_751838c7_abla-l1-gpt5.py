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
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"  # Output the video with detections and mAP text overlay
CONFIDENCE_THRESHOLD = 0.5

# EdgeTPU shared library path as per guideline
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


# =========================
# Utility Functions
# =========================
def load_labels(path):
    """
    Load label map file. Supports common formats:
    - Each line: "id label" or "id:label" or "label" (implicitly 0..N-1).
    Returns a dict {id: label_string}.
    """
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for idx, line in enumerate(lines):
        if ":" in line:
            parts = line.split(":", 1)
            try:
                key = int(parts[0].strip())
                labels[key] = parts[1].strip()
            except ValueError:
                labels[idx] = line.strip()
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[idx] = line.strip()
    return labels


def make_interpreter(model_path, delegate_lib):
    """
    Create a TFLite Interpreter with EdgeTPU delegate.
    """
    delegates = [load_delegate(delegate_lib)]
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame_bgr, input_details):
    """
    Preprocess frame for model input:
    - Resize to expected size
    - Convert BGR->RGB
    - Convert dtype as required
    - Add batch dimension
    Returns preprocessed input tensor and info dict.
    """
    h, w = input_details["shape"][1], input_details["shape"][2]
    dtype = input_details["dtype"]

    # Resize
    resized = cv2.resize(frame_bgr, (w, h))
    # Convert BGR -> RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if dtype == np.float32:
        input_tensor = rgb.astype(np.float32) / 255.0
    else:
        # Assume uint8 quantized input for EdgeTPU model
        input_tensor = rgb.astype(np.uint8)

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def get_output_indices(interpreter):
    """
    Determine indices for boxes, classes, scores, num_detections.
    Returns a dict with keys: boxes, classes, scores, count
    """
    output_details = interpreter.get_output_details()
    idx_map = {"boxes": None, "classes": None, "scores": None, "count": None}
    for od in output_details:
        shape = od["shape"]
        if len(shape) == 3 and shape[-1] == 4:
            idx_map["boxes"] = od["index"]
        elif len(shape) == 2:
            # Could be classes or scores
            # Try to infer by dtype: classes often float32 but integral semantics; rely on name if available
            # We instead defer: pick one for scores and one for classes by value ranges later if needed.
            # For robustness, we'll assign by "quantization" if present; otherwise use name heuristic.
            name = od.get("name", "").lower()
            if "score" in name or "scores" in name:
                idx_map["scores"] = od["index"]
            elif "class" in name or "classes" in name:
                idx_map["classes"] = od["index"]
            else:
                # Fallback: if dtype is float and not yet assigned scores, assume scores
                if od["dtype"] == np.float32 and idx_map["scores"] is None:
                    idx_map["scores"] = od["index"]
                else:
                    if idx_map["classes"] is None:
                        idx_map["classes"] = od["index"]
        elif len(shape) == 1 and shape[0] == 1:
            idx_map["count"] = od["index"]
    return idx_map


def draw_detections_on_frame(frame_bgr, detections, labels, map_text):
    """
    Draw bounding boxes and labels on the frame.
    detections: list of dicts with keys: bbox (xmin, ymin, xmax, ymax), score, class_id
    map_text: string to overlay for mAP value
    """
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        score = det["score"]
        class_id = det["class_id"]

        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

        # Label text
        label = labels.get(class_id, None)
        if label is None and (class_id + 1) in labels:
            label = labels.get(class_id + 1)  # Some label files are 1-based
        if label is None:
            label = f"id:{class_id}"

        text = f"{label} {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (int(xmin), int(ymin) - th - baseline), (int(xmin) + tw, int(ymin)), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (int(xmin), int(ymin) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Overlay mAP in top-left corner
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 10), 2)


def iou(box_a, box_b):
    """
    Compute IoU between two boxes in [xmin, ymin, xmax, ymax] absolute pixel format.
    """
    axmin, aymin, axmax, aymax = box_a
    bxmin, bymin, bxmax, bymax = box_b

    inter_xmin = max(axmin, bxmin)
    inter_ymin = max(aymin, bymin)
    inter_xmax = min(axmax, bxmax)
    inter_ymax = min(aymax, bymax)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area_a = max(0.0, axmax - axmin) * max(0.0, aymax - aymin)
    area_b = max(0.0, bxmax - bxmin) * max(0.0, bymax - bymin)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_map(predictions, ground_truth, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP) given predictions and ground truth.
    predictions: list of dicts with keys: frame, bbox [xmin,ymin,xmax,ymax], score, class_id
    ground_truth: dict mapping frame_index -> list of dicts {bbox, class_id}
    If ground_truth is None or empty, returns None.
    """
    if not ground_truth:
        return None

    # Organize GT by frame and class
    gt_by_frame = {}
    for frame_idx, items in ground_truth.items():
        gt_by_frame[frame_idx] = []
        for it in items:
            gt_by_frame[frame_idx].append({"bbox": it["bbox"], "class_id": it["class_id"], "matched": False})

    # Organize predictions by class
    classes = sorted(set([p["class_id"] for p in predictions]))
    ap_per_class = []

    for cls in classes:
        # Filter predictions of this class
        preds = [p for p in predictions if p["class_id"] == cls]
        # Sort by score descending
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        # Count GT of this class
        gt_count = 0
        for frame_idx, items in gt_by_frame.items():
            for it in items:
                if it["class_id"] == cls:
                    gt_count += 1

        if gt_count == 0:
            continue

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, pred in enumerate(preds):
            frame_idx = pred["frame"]
            pb = pred["bbox"]
            best_iou = 0.0
            best_gt_idx = -1

            gts = gt_by_frame.get(frame_idx, [])
            for j, gt in enumerate(gts):
                if gt["class_id"] != cls or gt["matched"]:
                    continue
                iou_val = iou(pb, gt["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1.0
                gt_by_frame[frame_idx][best_gt_idx]["matched"] = True
            else:
                fp[i] = 1.0

        # Precision-recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(gt_count)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        # AP: area under precision-recall curve (VOC2007 11-point or integral)
        # Use integration method
        # First, ensure precision is non-increasing w.r.t recall
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for k in range(mpre.size - 1, 0, -1):
            mpre[k - 1] = max(mpre[k - 1], mpre[k])
        # Identify points where recall changes
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        ap_per_class.append(ap)

    if not ap_per_class:
        return None
    return float(np.mean(ap_per_class))


# =========================
# Main Processing
# =========================
def main():
    # Validate paths
    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    if not os.path.isfile(INPUT_PATH):
        print(f"Input video not found: {INPUT_PATH}")
        return

    labels = load_labels(LABEL_PATH)

    # Setup interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_LIB)
    input_details_list = interpreter.get_input_details()
    if not input_details_list:
        print("Interpreter has no input details.")
        return
    input_details = {
        "index": input_details_list[0]["index"],
        "shape": input_details_list[0]["shape"],
        "dtype": input_details_list[0]["dtype"],
    }
    output_indices = get_output_indices(interpreter)

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Failed to open input video: {INPUT_PATH}")
        return

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 1e-2:
        in_fps = 30.0  # default fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (width, height))
    if not writer.isOpened():
        print(f"Failed to open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # For mAP computation
    predictions = []  # list of dicts: frame, bbox [xmin,ymin,xmax,ymax], score, class_id
    # No ground truth provided; mAP will be None -> display "N/A"
    map_text = "mAP: N/A (no ground truth)"

    total_frames = 0
    t_infer_sum = 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        total_frames += 1
        frame_h, frame_w = frame_bgr.shape[:2]

        # Preprocess
        input_tensor = preprocess_frame(frame_bgr, input_details)

        # Inference
        interpreter.set_tensor(input_details["index"], input_tensor)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000.0
        t_infer_sum += infer_ms

        # Gather outputs
        boxes = interpreter.get_tensor(output_indices["boxes"]) if output_indices["boxes"] is not None else None
        classes = interpreter.get_tensor(output_indices["classes"]) if output_indices["classes"] is not None else None
        scores = interpreter.get_tensor(output_indices["scores"]) if output_indices["scores"] is not None else None
        count = interpreter.get_tensor(output_indices["count"]) if output_indices["count"] is not None else None

        # Shape normalization
        if boxes is not None and boxes.ndim == 3:
            boxes = boxes[0]
        if classes is not None and classes.ndim == 2:
            classes = classes[0]
        if scores is not None and scores.ndim == 2:
            scores = scores[0]
        if count is not None and count.ndim == 1:
            num = int(count[0])
        else:
            num = boxes.shape[0] if boxes is not None else 0

        # Build detections list for current frame
        detections = []
        for i in range(num):
            if boxes is None or classes is None or scores is None:
                continue
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue

            # boxes from TFLite SSD are typically [ymin, xmin, ymax, xmax] normalized [0,1]
            y_min, x_min, y_max, x_max = boxes[i]
            xmin = max(0, int(x_min * frame_w))
            ymin = max(0, int(y_min * frame_h))
            xmax = min(frame_w - 1, int(x_max * frame_w))
            ymax = min(frame_h - 1, int(y_max * frame_h))

            class_id = int(classes[i])  # could be 0-based; label lookup handles both 0/1-based

            detections.append({
                "bbox": (xmin, ymin, xmax, ymax),
                "score": score,
                "class_id": class_id
            })

            # Accumulate for mAP computation (if GT were available)
            predictions.append({
                "frame": total_frames - 1,  # zero-based frame index
                "bbox": (xmin, ymin, xmax, ymax),
                "score": score,
                "class_id": class_id
            })

        # Draw detections and mAP text
        draw_detections_on_frame(frame_bgr, detections, labels, map_text)

        # Optionally, overlay inference time
        inf_text = f"Infer: {infer_ms:.1f} ms"
        cv2.putText(frame_bgr, inf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 200, 240), 2)

        # Write frame
        writer.write(frame_bgr)

    # Release resources
    cap.release()
    writer.release()

    # Compute mAP (if ground truth is provided; currently none)
    # ground_truth format expected (if available):
    # ground_truth = {
    #     frame_index: [
    #         {"bbox": (xmin,ymin,xmax,ymax), "class_id": int}, ...
    #     ],
    #     ...
    # }
    ground_truth = None
    mAP = compute_map(predictions, ground_truth, iou_threshold=0.5)
    if mAP is None:
        print("mAP: N/A (no ground truth provided).")
    else:
        print(f"mAP@0.5: {mAP:.4f}")

    if total_frames > 0:
        avg_infer_ms = t_infer_sum / total_frames
        print(f"Processed {total_frames} frames. Avg inference: {avg_infer_ms:.2f} ms/frame")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()