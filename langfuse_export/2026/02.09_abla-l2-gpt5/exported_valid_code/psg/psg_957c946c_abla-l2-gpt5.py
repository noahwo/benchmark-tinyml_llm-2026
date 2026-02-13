import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP text
CONFIDENCE_THRESHOLD = 0.5

# -----------------------------
# Utility functions
# -----------------------------
def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def ensure_dir_for_file(filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def preprocess_frame_bgr_to_model_input(frame_bgr, input_details):
    # Determine expected input shape and dtype
    input_shape = input_details[0]['shape']  # [1, h, w, c]
    input_dtype = input_details[0]['dtype']
    _, in_h, in_w, _ = input_shape

    # Convert BGR to RGB and resize
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    # Prepare input tensor with quantization handling
    tensor = np.expand_dims(frame_resized, axis=0)

    # Handle different dtypes and quantization
    quant = input_details[0].get('quantization', (0.0, 0))
    scale, zero_point = quant if isinstance(quant, (tuple, list)) else (0.0, 0)

    if input_dtype == np.uint8:
        # Most common path for SSD MobileNet v1 (quantized uint8)
        tensor = tensor.astype(np.uint8)
    elif input_dtype == np.int8:
        # Quantize uint8 image into int8 domain using provided scale/zero_point
        tensor = tensor.astype(np.float32)
        if scale == 0:
            # Fallback: center around 0 before casting; typical but not guaranteed
            tensor = (tensor - 128.0).astype(np.int8)
        else:
            tensor = np.clip(np.round(tensor / scale + zero_point), -128, 127).astype(np.int8)
    else:
        # float input
        tensor = tensor.astype(np.float32) / 255.0

    return tensor

def get_tflite_outputs(interpreter):
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    # Dequantize outputs if needed and detect which is boxes/scores/classes/num_detections
    def dequantize(arr, od):
        quant = od.get('quantization', (0.0, 0))
        scale, zero_point = quant if isinstance(quant, (tuple, list)) else (0.0, 0)
        if scale and (od['dtype'] != np.float32):
            return (arr.astype(np.float32) - float(zero_point)) * float(scale)
        return arr

    outs = [dequantize(arr, od) for arr, od in zip(outputs, output_details)]

    # Identify tensors by shape/dtype heuristics
    boxes = None
    classes = None
    scores = None
    num = None
    for arr in outs:
        shp = arr.shape
        if len(shp) == 3 and shp[-1] == 4:
            boxes = arr
        elif len(shp) == 2:
            # Could be classes or scores [1, N]
            if arr.dtype.kind in ('f',):  # float -> likely scores
                scores = arr
            else:
                classes = arr
        elif len(shp) == 1 and shp[0] == 1:
            num = arr
        elif len(shp) == 2 and shp[-1] == 1 and shp[0] == 1:
            num = arr

    # Some models output classes as float but representing integers; cast to int
    if classes is not None and classes.dtype.kind == 'f':
        classes = classes.astype(np.int32)

    return boxes, classes, scores, num

def draw_detections(frame_bgr, boxes, classes, scores, labels, conf_thresh):
    h, w = frame_bgr.shape[:2]
    if boxes is None or classes is None or scores is None:
        return frame_bgr, []

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    # Ensure arrays are 1D aligned
    if len(boxes.shape) == 1 and boxes.shape[0] == 4:
        boxes = boxes[np.newaxis, :]
    if classes.ndim == 0:
        classes = np.array([int(classes)])
    if scores.ndim == 0:
        scores = np.array([float(scores)])

    detections_for_frame = []

    num_dets = min(len(scores), len(boxes), len(classes))
    for i in range(num_dets):
        score = float(scores[i])
        if score < conf_thresh:
            continue

        # boxes typically in [ymin, xmin, ymax, xmax] normalized
        y_min, x_min, y_max, x_max = boxes[i]
        x1 = int(max(0, min(w - 1, round(x_min * w))))
        y1 = int(max(0, min(h - 1, round(y_min * h))))
        x2 = int(max(0, min(w - 1, round(x_max * w))))
        y2 = int(max(0, min(h - 1, round(y_max * h))))

        class_id = int(classes[i])
        label = labels[class_id] if (0 <= class_id < len(labels)) else f"id:{class_id}"
        color = (0, 255, 0)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytext = max(0, y1 - 5)
        cv2.rectangle(frame_bgr, (x1, ytext - th - 4), (x1 + tw + 4, ytext + 2), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (x1 + 2, ytext), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        detections_for_frame.append({
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "class_id": class_id,
            "label": label
        })

    return frame_bgr, detections_for_frame

def compute_map(detections_per_frame, ground_truth_per_frame=None, iou_threshold=0.5):
    # Proper mAP requires ground-truth annotations; if not provided, return None.
    if ground_truth_per_frame is None or len(ground_truth_per_frame) == 0:
        return None

    # Example structure for ground_truth_per_frame (not provided in config):
    # ground_truth_per_frame = [
    #   [ {"bbox":[x1,y1,x2,y2], "class_id":int}, ... ],   # frame 0
    #   [ {"bbox":[x1,y1,x2,y2], "class_id":int}, ... ],   # frame 1
    #   ...
    # ]

    # Build per-class lists
    det_by_cls = {}
    gt_by_cls = {}

    for idx, dets in enumerate(detections_per_frame):
        for d in dets:
            c = d["class_id"]
            det_by_cls.setdefault(c, []).append((idx, d["score"], np.array(d["bbox"], dtype=np.float32)))
    for idx, gts in enumerate(ground_truth_per_frame):
        for g in gts:
            c = g["class_id"]
            gt_by_cls.setdefault(c, []).append((idx, np.array(g["bbox"], dtype=np.float32)))

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1)
        areaA = max(0.0, (boxA[2] - boxA[0] + 1)) * max(0.0, (boxA[3] - boxA[1] + 1))
        areaB = max(0.0, (boxB[2] - boxB[0] + 1)) * max(0.0, (boxB[3] - boxB[1] + 1))
        denom = areaA + areaB - inter
        return inter / denom if denom > 0 else 0.0

    aps = []
    for c in sorted(set(list(det_by_cls.keys()) + list(gt_by_cls.keys()))):
        dets = det_by_cls.get(c, [])
        gts = gt_by_cls.get(c, [])

        # Map of frame_idx -> list of GT boxes and matched flags
        gt_map = {}
        for frame_idx, gt_box in gts:
            gt_map.setdefault(frame_idx, []).append({"box": gt_box, "matched": False})

        # Sort detections by score descending
        dets_sorted = sorted(dets, key=lambda x: x[1], reverse=True)

        tp = []
        fp = []
        for frame_idx, score, dbox in dets_sorted:
            matched = False
            if frame_idx in gt_map:
                ious = [iou(dbox, g["box"]) for g in gt_map[frame_idx]]
                if len(ious) > 0:
                    best_idx = int(np.argmax(ious))
                    if ious[best_idx] >= iou_threshold and not gt_map[frame_idx][best_idx]["matched"]:
                        matched = True
                        gt_map[frame_idx][best_idx]["matched"] = True
            tp.append(1 if matched else 0)
            fp.append(0 if matched else 1)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        npos = sum(len(v) for v in gt_map.values())
        if npos == 0:
            continue
        rec = tp_cum / float(npos)
        prec = np.divide(tp_cum, (tp_cum + fp_cum + 1e-9))

        # Compute AP as area under precision envelope
        # VOC-style continuous interpolation
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# -----------------------------
# Main pipeline (PROGRAMMING GUIDELINE)
# 1) Setup, load interpreter, allocate tensors, load labels, open input video
# 2) Preprocessing
# 3) Inference
# 4) Output handling (draw boxes, labels, compute mAP, save video)
# -----------------------------
def main():
    ensure_dir_for_file(OUTPUT_PATH)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    # Prepare video writer after reading first frame to get frame size
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Failed to read first frame from input video.")

    h, w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # For storing detections to (optionally) compute mAP (requires GT which is not provided)
    detections_per_frame = []

    # Process first frame then loop
    total_frames = 0
    inf_times = []
    map_text = "mAP: N/A (no ground truth provided)"
    start_time_overall = time.time()

    def process_and_write(frame):
        nonlocal total_frames

        # Preprocessing (2)
        input_tensor = preprocess_frame_bgr_to_model_input(frame, input_details)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)

        # Inference (3)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        inf_times.append(t1 - t0)

        # Extract outputs
        boxes, classes, scores, num = get_tflite_outputs(interpreter)

        # Output handling: draw detections (4)
        annotated, dets = draw_detections(frame, boxes, classes, scores, labels, CONFIDENCE_THRESHOLD)
        detections_per_frame.append(dets)

        # Overlay mAP text (static since GT not available)
        cv2.putText(annotated, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2, cv2.LINE_AA)

        writer.write(annotated)
        total_frames += 1

    # Process the first frame
    process_and_write(first_frame)

    # Process remaining frames
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        process_and_write(frame)

    # Release resources for video writing
    writer.release()
    cap.release()

    elapsed = time.time() - start_time_overall
    avg_inf_ms = (np.mean(inf_times) * 1000.0) if inf_times else 0.0

    # Compute mAP after processing (requires ground truth; not provided)
    mAP_value = compute_map(detections_per_frame, ground_truth_per_frame=None)
    if mAP_value is None:
        print("mAP: N/A (no ground truth provided). Video saved with detection overlays.")
    else:
        print(f"Computed mAP: {mAP_value:.4f}")

    print(f"Processed frames: {total_frames}")
    print(f"Total elapsed time: {elapsed:.2f}s, Avg inference time: {avg_inf_ms:.2f} ms/frame")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()