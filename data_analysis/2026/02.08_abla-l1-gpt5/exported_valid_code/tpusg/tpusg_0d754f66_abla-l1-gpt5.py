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
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGE_TPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# =========================
# Helpers
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Support two common formats:
            # 1) "0 person"
            # 2) "person" (implicit incremental index)
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                idx = int(parts[0])
                labels[idx] = parts[1]
            else:
                labels[i] = line
    return labels

def set_input_tensor(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    _, height, width, _ = input_details["shape"]
    input_dtype = input_details["dtype"]

    # Preprocess: BGR to RGB, resize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (width, height))

    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)

    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details["index"], input_data)

def get_output_tensors(interpreter):
    output_details = interpreter.get_output_details()
    tensors = [interpreter.get_tensor(d["index"]) for d in output_details]

    boxes = None
    classes = None
    scores = None
    count = None

    # Heuristic mapping by name/shape to handle common detection models
    for d, t in zip(output_details, tensors):
        name = d.get("name", "").lower()
        shape = d.get("shape", [])
        if len(shape) == 3 and shape[-1] == 4:
            boxes = t
        elif len(shape) == 2:
            if "score" in name:
                scores = t
            elif "class" in name:
                classes = t
            else:
                # If names are not clear, assign by elimination later
                if classes is None:
                    classes = t
                else:
                    scores = t
        elif len(shape) == 1 and shape[0] == 1:
            count = int(np.squeeze(t))

    # Fallback if count wasn't explicitly provided
    if count is None and scores is not None:
        count = scores.shape[-1]

    # Squeeze leading batch dimension if present
    if boxes is not None and boxes.ndim == 3:
        boxes = boxes[0]
    if classes is not None and classes.ndim == 2:
        classes = classes[0]
    if scores is not None and scores.ndim == 2:
        scores = scores[0]

    return boxes, classes, scores, count

def to_pixel_coords(box, img_w, img_h):
    # box in [ymin, xmin, ymax, xmax] normalized
    ymin, xmin, ymax, xmax = box
    left = int(max(0, xmin * img_w))
    top = int(max(0, ymin * img_h))
    right = int(min(img_w - 1, xmax * img_w))
    bottom = int(min(img_h - 1, ymax * img_h))
    return left, top, right, bottom

def iou_xyxy(box_a, box_b):
    # boxes as [x1, y1, x2, y2]
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom

def nms_xyxy(boxes, scores, iou_thresh=0.5):
    # boxes: Nx4 (x1,y1,x2,y2), scores: N
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    boxes = np.array(boxes)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        masked = rest[ious <= iou_thresh]
        order = masked
    return keep

def average_precision(scores_sorted, is_tp_sorted):
    # scores_sorted: list/array of scores sorted descending
    # is_tp_sorted: list/array of booleans aligned to scores_sorted
    scores_sorted = np.asarray(scores_sorted, dtype=np.float32)
    is_tp_sorted = np.asarray(is_tp_sorted, dtype=np.bool_)
    num_dets = len(scores_sorted)
    num_gt = int(np.sum(is_tp_sorted))
    if num_dets == 0 or num_gt == 0:
        return None  # Not computable for this class/sample

    cum_tp = np.cumsum(is_tp_sorted.astype(np.float32))
    cum_fp = np.cumsum((~is_tp_sorted).astype(np.float32))
    precisions = cum_tp / (cum_tp + cum_fp + 1e-12)
    recalls = cum_tp / (num_gt + 1e-12)

    # Interpolated AP (VOC 2010+ style)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def draw_detections(frame, detections, labels):
    # detections: list of dicts with keys: bbox(x1,y1,x2,y2), class_id, score
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        score = det["score"]
        label = labels.get(int(cls_id), str(int(cls_id)))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, caption, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def overlay_info(frame, text, origin=(8, 24)):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

# =========================
# Main Pipeline
# =========================
def main():
    # Validate paths
    for pth, desc in [(MODEL_PATH, "Model"), (LABEL_PATH, "Label map"), (INPUT_PATH, "Input video")]:
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"{desc} not found at: {pth}")

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter with EdgeTPU
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate(EDGE_TPU_LIB)]
    )
    interpreter.allocate_tensors()

    # Prepare video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0  # fallback

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # For mAP aggregation (proxy mAP based on NMS-as-GT per frame)
    ap_sums = {}   # class_id -> sum of AP across frames where class present
    ap_counts = {} # class_id -> count of frames where AP was computed (i.e., num_gt > 0)
    running_map = 0.0
    frames_processed = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1
        img_h, img_w = frame.shape[:2]

        # Preprocess and inference
        set_input_tensor(interpreter, frame)
        interpreter.invoke()
        boxes_norm, classes, scores, count = get_output_tensors(interpreter)

        detections = []
        if boxes_norm is not None and classes is not None and scores is not None and count is not None:
            n = int(count)
            for i in range(n):
                score = float(scores[i])
                if score < CONFIDENCE_THRESHOLD:
                    continue
                cls_id = int(classes[i])
                x1, y1, x2, y2 = to_pixel_coords(boxes_norm[i], img_w, img_h)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "score": score
                })

        # Draw detections
        draw_detections(frame, detections, labels)

        # Compute per-frame AP per class (proxy: NMS survivors as "GT", suppressed as FP)
        # Group detections by class
        by_class = {}
        for idx, det in enumerate(detections):
            cid = det["class_id"]
            by_class.setdefault(cid, []).append(idx)

        per_class_ap = {}
        for cid, idxs in by_class.items():
            if not idxs:
                continue
            # Sort by score descending
            idxs_sorted = sorted(idxs, key=lambda i: detections[i]["score"], reverse=True)
            boxes_c = [detections[i]["bbox"] for i in idxs_sorted]
            scores_c = [detections[i]["score"] for i in idxs_sorted]

            # NMS (on sorted lists). The indices returned are relative to sorted list.
            keep_sorted_positions = nms_xyxy(boxes_c, scores_c, iou_thresh=0.5)
            is_tp_sorted = np.zeros(len(idxs_sorted), dtype=bool)
            if keep_sorted_positions:
                is_tp_sorted[keep_sorted_positions] = True

            ap = average_precision(scores_c, is_tp_sorted)
            if ap is not None:
                per_class_ap[cid] = ap
                ap_sums[cid] = ap_sums.get(cid, 0.0) + ap
                ap_counts[cid] = ap_counts.get(cid, 0) + 1

        # Compute running mAP across classes encountered so far
        valid_classes = [cid for cid, cnt in ap_counts.items() if cnt > 0]
        if valid_classes:
            class_means = [ap_sums[cid] / ap_counts[cid] for cid in valid_classes]
            running_map = float(np.mean(class_means))
        else:
            running_map = 0.0

        # Overlay information
        overlay_info(frame, f"TFLite object detection (TPU) | mAP (proxy): {running_map:.3f}")
        # Optional: overlay FPS
        elapsed = time.time() - start_time
        fps_inst = frames_processed / max(elapsed, 1e-6)
        overlay_info(frame, f"FPS: {fps_inst:.1f}", origin=(8, 50))

        # Write frame
        writer.write(frame)

    # Release resources
    cap.release()
    writer.release()

    # Final report
    valid_classes = [cid for cid, cnt in ap_counts.items() if cnt > 0]
    if valid_classes:
        final_map = float(np.mean([ap_sums[cid] / ap_counts[cid] for cid in valid_classes]))
    else:
        final_map = 0.0

    print("Processing completed.")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frames_processed}")
    print(f"Proxy mAP over video (NMS-as-GT): {final_map:.4f}")

if __name__ == "__main__":
    main()