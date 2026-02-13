import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(path):
    labels = {}
    try:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        for idx, line in enumerate(lines):
            # Support formats like "0 person" or just "person"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
            else:
                labels[idx] = line
    except Exception:
        pass
    return labels

def initialize_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Assume single input tensor
    in_shape = input_details[0]['shape']
    if len(in_shape) != 4:
        raise RuntimeError("Expected input tensor of rank 4, got: {}".format(in_shape))
    in_height, in_width = int(in_shape[1]), int(in_shape[2])
    return interpreter, input_details, output_details, (in_width, in_height)

def preprocess_frame(frame_bgr, input_details, input_size):
    in_w, in_h = input_size
    # Convert BGR to RGB as most TFLite detection models expect RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h))
    input_tensor = np.expand_dims(resized, axis=0)
    tensor_info = input_details[0]
    dtype = tensor_info['dtype']

    # Handle quantization and dtype
    if dtype == np.float32:
        input_tensor = input_tensor.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        # Many uint8 TFLite models take raw 0-255 inputs directly
        input_tensor = input_tensor.astype(np.uint8)
    elif dtype == np.int8:
        # Quantized int8: map float [0,1] to quantized using scale/zero_point
        scale, zero_point = tensor_info.get('quantization', (0.0, 0))
        if scale is None or scale == 0.0:
            # Fallback: just cast (might be incorrect for a given model)
            input_tensor = input_tensor.astype(np.int8)
        else:
            real = (input_tensor.astype(np.float32) / 255.0)
            quant = np.round(real / scale + zero_point)
            quant = np.clip(quant, -128, 127).astype(np.int8)
            input_tensor = quant
    else:
        # Fallback: attempt to cast to required dtype
        input_tensor = input_tensor.astype(dtype)

    return input_tensor

def parse_detections(interpreter, output_details, frame_shape):
    """
    Parse outputs of a TFLite SSD model. Returns:
    - boxes_px: (N, 4) array of [xmin, ymin, xmax, ymax] in pixel coords
    - classes: (N,) int array of class indices
    - scores: (N,) float array of confidence scores
    It attempts to be robust to different output orders.
    """
    outs = []
    for d in output_details:
        outs.append(interpreter.get_tensor(d['index']))

    # Squeeze outputs for easier handling
    squeezed = [np.squeeze(o) for o in outs]

    boxes = None
    classes = None
    scores = None
    num = None

    # Identify boxes tensor: 2D with last dim 4
    for arr in squeezed:
        if arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = arr  # shape (N,4) with [ymin, xmin, ymax, xmax]
            break

    # Identify num_detections: 0D or 1-element
    for arr in squeezed:
        if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
            try:
                num = int(arr.reshape(()))
            except Exception:
                num = None
            break

    # Identify classes and scores among 1D arrays
    candidates = [arr for arr in squeezed if arr.ndim == 1 and (boxes is None or arr.size == boxes.shape[0])]
    # Heuristics: scores are float in [0,1], classes are near-integers
    cand_scores = None
    cand_classes = None
    for arr in candidates:
        if arr.dtype.kind in ('f',):  # float
            if np.all((arr >= 0.0) & (arr <= 1.0)):
                cand_scores = arr
            else:
                # Could still be classes as float
                if np.allclose(arr, np.round(arr)):
                    cand_classes = arr.astype(np.int32)
        else:
            # int types likely classes
            cand_classes = arr.astype(np.int32)

    # If still unresolved, attempt to decide by variance
    if cand_scores is None and candidates:
        # Pick the one with values between 0 and 1 if possible, otherwise the one with greater variance as scores
        for arr in candidates:
            if np.all((arr >= 0.0) & (arr <= 1.0)):
                cand_scores = arr
                break
        if cand_scores is None:
            # Fallback: pick float array as scores
            floats = [a for a in candidates if a.dtype.kind in ('f',)]
            if floats:
                cand_scores = floats[0]
    if cand_classes is None and candidates:
        # Pick the remaining as classes (rounded)
        remaining = [a for a in candidates if a is not cand_scores]
        if remaining:
            cand_classes = np.round(remaining[0]).astype(np.int32)

    scores = cand_scores
    classes = cand_classes

    # If num is provided, clamp arrays to num
    if boxes is None or classes is None or scores is None:
        raise RuntimeError("Unable to parse detection outputs. Got shapes: {}".format([a.shape for a in squeezed]))
    N = boxes.shape[0]
    if num is not None:
        N = min(N, int(num))
    boxes = boxes[:N]
    classes = classes[:N]
    scores = scores[:N]

    # Convert normalized [ymin, xmin, ymax, xmax] to pixel coords [xmin, ymin, xmax, ymax]
    h, w = int(frame_shape[0]), int(frame_shape[1])
    ymin = boxes[:, 0] * h
    xmin = boxes[:, 1] * w
    ymax = boxes[:, 2] * h
    xmax = boxes[:, 3] * w
    boxes_px = np.stack([
        np.clip(xmin, 0, w - 1),
        np.clip(ymin, 0, h - 1),
        np.clip(xmax, 0, w - 1),
        np.clip(ymax, 0, h - 1)
    ], axis=1).astype(np.int32)

    return boxes_px, classes.astype(np.int32), scores.astype(np.float32)

def deterministic_color(class_id):
    rng = np.random.RandomState(class_id * 9973 + 12345)
    color = tuple(int(x) for x in rng.randint(0, 255, size=3))
    return color

def compute_pseudo_map(detections_per_frame, num_classes):
    """
    Compute a proxy mAP using only predictions (no ground truth available).
    For each class, in each frame we treat the highest-score detection as a TP,
    and additional detections of the same class in that frame as FPs. This
    yields a well-defined Average Precision per class and their mean (mAP).
    """
    APs = []
    for c in range(num_classes):
        # Collect per-frame detections for class c
        entries = []  # list of (score, is_tp)
        positive_frames = 0
        for det in detections_per_frame:
            classes = det['classes']
            scores = det['scores']
            # Indices for class c
            idxs = np.where(classes == c)[0]
            if idxs.size == 0:
                continue
            # Mark highest score in this frame for this class as TP
            frame_scores = scores[idxs]
            top_idx_local = int(np.argmax(frame_scores))
            top_global_idx = idxs[top_idx_local]
            positive_frames += 1
            for j, gi in enumerate(idxs):
                is_tp = (gi == top_global_idx)
                entries.append((float(scores[gi]), 1 if is_tp else 0))

        if positive_frames == 0 or len(entries) == 0:
            continue

        # Sort by score descending
        entries.sort(key=lambda x: -x[0])
        tps = np.array([e[1] for e in entries], dtype=np.float32)
        fps = 1.0 - tps

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        recall = cum_tp / max(positive_frames, 1)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

        # Compute AP using the standard interpolated precision envelope
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        # Sum over recall steps where it changes
        i_changes = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[i_changes + 1] - mrec[i_changes]) * mpre[i_changes + 1]))
        APs.append(ap)

    if len(APs) == 0:
        return None
    return float(np.mean(APs))

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    # Load labels
    labels_dict = load_labels(LABEL_PATH)
    max_label_id = max(labels_dict.keys()) if labels_dict else -1
    num_classes = max_label_id + 1 if max_label_id >= 0 else 91  # default to COCO-ish if unknown

    # Initialize interpreter
    interpreter, input_details, output_details, input_size = initialize_interpreter(MODEL_PATH)

    # Prepare video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video: {}".format(INPUT_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # First pass: run inference and collect detections for all frames
    detections_per_frame = []
    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess_frame(frame, input_details, input_size)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        boxes_px, classes, scores = parse_detections(interpreter, output_details, frame.shape[:2])
        # Store raw detections for further processing and drawing later
        detections_per_frame.append({
            'boxes': boxes_px,
            'classes': classes,
            'scores': scores
        })
        frame_idx += 1
        if frame_idx % 50 == 0:
            if total_frames > 0:
                print("Inference pass: processed {}/{} frames...".format(frame_idx, total_frames))
            else:
                print("Inference pass: processed {} frames...".format(frame_idx))
    cap.release()
    t1 = time.time()
    elapsed_infer = t1 - t0
    print("Inference completed on {} frames in {:.2f}s ({:.2f} FPS)".format(
        frame_idx, elapsed_infer, frame_idx / max(elapsed_infer, 1e-6)))

    # Compute pseudo-mAP from detections (proxy metric without ground truth)
    map_value = compute_pseudo_map(detections_per_frame, num_classes)
    if map_value is None:
        map_text = "mAP: N/A"
    else:
        map_text = "mAP: {:.3f}".format(map_value)

    # Second pass: draw boxes above threshold and write output video with mAP overlay
    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open output writer: {}".format(OUTPUT_PATH))

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to reopen input video: {}".format(INPUT_PATH))

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 0.6

    frame_idx = 0
    t2 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(detections_per_frame):
            break
        det = detections_per_frame[frame_idx]
        boxes = det['boxes']
        classes = det['classes']
        scores = det['scores']

        # Draw detections above threshold
        for i in range(len(scores)):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(classes[i])
            label = labels_dict.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            color = deterministic_color(cls_id)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text
            text = "{} {:.2f}".format(label, score)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            y_text = max(y1 - 8, th + 4)
            x_text = max(x1, 0)
            # Filled background for readability
            cv2.rectangle(frame, (x_text, y_text - th - 4), (x_text + tw + 2, y_text + 2), color, -1)
            cv2.putText(frame, text, (x_text + 1, y_text - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Overlay mAP
        cv2.putText(frame, map_text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print("Writing pass: wrote {} frames...".format(frame_idx))

    cap.release()
    writer.release()
    t3 = time.time()
    print("Output saved to: {}".format(OUTPUT_PATH))
    print("Writing completed in {:.2f}s".format(t3 - t2))

if __name__ == "__main__":
    main()