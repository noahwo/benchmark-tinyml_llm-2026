import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# ----------------------------
# Configuration parameters
# ----------------------------
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
PSEUDO_GT_CONFIDENCE = 0.7  # threshold used to derive pseudo ground truth from detections
NMS_IOU_THRESHOLD = 0.5     # IoU threshold for NMS and matching
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# ----------------------------
# Utility functions
# ----------------------------
def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Support formats:
            # "0 person", "0: person", or "person"
            parts = line.replace(":", " ").split()
            if parts and parts[0].isdigit():
                idx = int(parts[0])
                name = " ".join(parts[1:]).strip()
                if not name:
                    name = str(idx)
                labels[idx] = name
            else:
                labels[i] = line
    return labels

def make_interpreter(model_path):
    return Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(EDGETPU_LIB)]
    )

def preprocess_frame(frame_bgr, input_size, input_dtype):
    h_in, w_in = input_size
    # Convert BGR (OpenCV) to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_in, h_in))
    if input_dtype == np.uint8:
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
    else:
        # Fallback for float models
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return input_data

def get_output(interpreter):
    # Assumes standard TFLite SSD output: boxes, classes, scores, count
    output_details = interpreter.get_output_details()
    tensors = [interpreter.get_tensor(od['index']) for od in output_details]

    # Attempt to map outputs robustly
    boxes = None
    classes = None
    scores = None
    count = None

    # Heuristic mapping
    for t in tensors:
        arr = np.squeeze(t)
        if arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = arr  # [N,4] in [ymin,xmin,ymax,xmax] normalized
        elif arr.ndim == 1 and arr.size == 1:
            count = int(arr[0])
        elif arr.ndim in (1, 2):
            # candidates for classes or scores
            if arr.ndim == 2:
                arr = arr[0]
            # defer assignment, we'll decide after we see both
            if classes is None:
                classes = arr
            elif scores is None:
                scores = arr

    # If classes/scores are swapped, fix by checking ranges
    if classes is not None and scores is not None:
        classes_is_int_like = np.all((classes % 1) == 0)
        scores_in_01 = np.all((scores >= 0.0) & (scores <= 1.0))
        classes_in_01 = np.all((classes >= 0.0) & (classes <= 1.0))
        if not scores_in_01 and classes_in_01:
            # swap
            classes, scores = scores, classes
    # Default fallback to common order if anything missing
    if boxes is None or scores is None or classes is None:
        # Try strict common order
        try:
            boxes = tensors[0][0]
            classes = tensors[1][0]
            scores = tensors[2][0]
            count = int(tensors[3][0])
        except Exception:
            raise RuntimeError("Unable to parse TFLite detection outputs. Unexpected tensor shapes.")

    if count is None:
        count = len(scores)
    count = int(count)
    boxes = boxes[:count]
    classes = classes[:count]
    scores = scores[:count]
    return boxes, classes, scores, count

def to_pixel_boxes(norm_boxes, img_w, img_h):
    # norm_boxes: [N,4] in [ymin,xmin,ymax,xmax], normalized 0..1
    pixel_boxes = []
    for b in norm_boxes:
        y1 = max(0, min(img_h, int(b[0] * img_h)))
        x1 = max(0, min(img_w, int(b[1] * img_w)))
        y2 = max(0, min(img_h, int(b[2] * img_h)))
        x2 = max(0, min(img_w, int(b[3] * img_w)))
        # ensure y1<=y2, x1<=x2
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        pixel_boxes.append((x1, y1, x2, y2))
    return pixel_boxes

def compute_iou(boxA, boxB):
    # boxes as (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = areaA + areaB - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def nms_per_class(detections, iou_thresh=0.5):
    # detections: list of dicts with keys: bbox, score, class_id
    result = []
    # group by class
    by_class = {}
    for d in detections:
        by_class.setdefault(d['class_id'], []).append(d)
    for cid, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda x: x['score'], reverse=True)
        keep = []
        for d in dets_sorted:
            should_keep = True
            for k in keep:
                if compute_iou(d['bbox'], k['bbox']) >= iou_thresh:
                    should_keep = False
                    break
            if should_keep:
                keep.append(d)
        result.extend(keep)
    return result

def color_for_class(cid):
    # deterministic color based on class id
    np.random.seed(cid + 12345)
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return color

def draw_detections(frame, detections, labels, map_text=None):
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cid = d['class_id']
        score = d['score']
        label = labels.get(cid, str(cid))
        color = color_for_class(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    if map_text is not None:
        cv2.putText(frame, map_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

def compute_ap(recalls, precisions):
    # VOC-style AP with interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def compute_map(all_preds_by_class, all_gts_by_class, iou_thresh=0.5):
    aps = []
    for cid in all_gts_by_class.keys():
        gts = all_gts_by_class.get(cid, [])
        preds = all_preds_by_class.get(cid, [])
        if len(gts) == 0 or len(preds) == 0:
            continue

        # Organize GTs per frame and track matched flags
        gts_by_frame = {}
        for fr, box in gts:
            gts_by_frame.setdefault(fr, [])
            gts_by_frame[fr].append({'bbox': box, 'matched': False})

        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)  # (score, frame_idx, box)
        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        total_gt = sum(len(lst) for lst in gts_by_frame.values())
        if total_gt == 0:
            continue

        for i, (score, fr, pbox) in enumerate(preds_sorted):
            gtcands = gts_by_frame.get(fr, [])
            iou_max = 0.0
            jmax = -1
            for j, g in enumerate(gtcands):
                if g['matched']:
                    continue
                iou = compute_iou(pbox, g['bbox'])
                if iou > iou_max:
                    iou_max = iou
                    jmax = j
            if iou_max >= iou_thresh and jmax >= 0:
                if not gtcands[jmax]['matched']:
                    tp[i] = 1.0
                    gtcands[jmax]['matched'] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Precision-recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(1, total_gt)
        precisions = cum_tp / np.maximum(1e-9, (cum_tp + cum_fp))
        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    if len(aps) == 0:
        return None
    return float(np.mean(aps))

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite EdgeTPU interpreter
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    # Input tensor details
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    _, in_h, in_w, _ = input_details[0]['shape']
    in_dtype = input_details[0]['dtype']

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # For pseudo mAP computation across frames
    all_preds_by_class = {}  # cid -> list of (score, frame_idx, box)
    all_gts_by_class = {}    # cid -> list of (frame_idx, box)

    frame_idx = 0
    t0 = time.time()
    infer_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            input_data = preprocess_frame(frame, (in_h, in_w), in_dtype)
            interpreter.set_tensor(input_index, input_data)

            # Inference
            t_infer0 = time.time()
            interpreter.invoke()
            t_infer1 = time.time()
            infer_times.append(t_infer1 - t_infer0)

            # Postprocess
            boxes_n, classes_n, scores_n, count = get_output(interpreter)
            pixel_boxes = to_pixel_boxes(boxes_n, width, height)

            # Build detections list
            detections = []
            for i in range(count):
                score = float(scores_n[i])
                cid = int(classes_n[i])
                bbox = pixel_boxes[i]
                detections.append({'bbox': bbox, 'score': score, 'class_id': cid})

            # NMS for drawing and thresholding
            dets_nms = nms_per_class([d for d in detections if d['score'] >= CONFIDENCE_THRESHOLD],
                                     iou_thresh=NMS_IOU_THRESHOLD)

            # Update pseudo ground truth and prediction pools for mAP
            # Pseudo-GT: high confidence detections (after NMS)
            gt_candidates = nms_per_class([d for d in detections if d['score'] >= PSEUDO_GT_CONFIDENCE],
                                          iou_thresh=NMS_IOU_THRESHOLD)
            for d in gt_candidates:
                cid = d['class_id']
                all_gts_by_class.setdefault(cid, []).append((frame_idx, d['bbox']))

            # Predictions: use all detections (model is already NMS-ed), add to pools
            for d in detections:
                cid = d['class_id']
                all_preds_by_class.setdefault(cid, []).append((d['score'], frame_idx, d['bbox']))

            # Compute running mAP (pseudo, due to lack of true GT)
            running_map = compute_map(all_preds_by_class, all_gts_by_class, iou_thresh=NMS_IOU_THRESHOLD)
            map_text = f"mAP (pseudo): {running_map:.3f}" if running_map is not None else "mAP (pseudo): N/A"

            # Draw and write
            draw_detections(frame, dets_nms, labels, map_text=map_text)
            writer.write(frame)

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    avg_infer = (sum(infer_times) / len(infer_times)) if infer_times else 0.0
    final_map = compute_map(all_preds_by_class, all_gts_by_class, iou_thresh=NMS_IOU_THRESHOLD)
    print("Processing complete.")
    print(f"Frames processed: {frame_idx}")
    print(f"Total time: {total_time:.2f}s, FPS: {frame_idx / total_time:.2f}")
    print(f"Average inference time: {avg_infer * 1000:.2f} ms")
    if final_map is not None:
        print(f"Final pseudo mAP: {final_map:.3f}")
    else:
        print("Final pseudo mAP: N/A (insufficient pseudo ground truth)")

if __name__ == "__main__":
    main()