import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"         # Read a single video file from the given input_path
output_path = "results/object_detection/test_results/sheeps_detections.mp4"  # Output video with boxes, labels, and mAP
confidence_threshold = 0.5

# =========================
# Utility Functions
# =========================
def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def class_id_to_label(class_id_raw, labels):
    # Try both 0-based and 1-based indexing robustly
    ci = int(class_id_raw)
    if 0 <= ci < len(labels):
        return labels[ci]
    if 1 <= ci <= len(labels):
        return labels[ci - 1]
    return f"class_{ci}"

def get_color_for_class(class_id):
    # Deterministic pseudo-random color from class id
    rng = np.random.default_rng(seed=int(class_id) + 12345)
    color = rng.integers(0, 255, size=3, dtype=np.uint8).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def draw_labelled_box(frame, box, label, score, color):
    h, w = frame.shape[:2]
    ymin, xmin, ymax, xmax = box
    x1 = max(0, min(int(xmin * w), w - 1))
    y1 = max(0, min(int(ymin * h), h - 1))
    x2 = max(0, min(int(xmax * w), w - 1))
    y2 = max(0, min(int(ymax * h), h - 1))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_text = f"{label}: {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(0, y1 - th - baseline)
    cv2.rectangle(frame, (x1, y_text), (x1 + tw + 2, y_text + th + baseline + 2), color, -1)
    cv2.putText(frame, label_text, (x1 + 1, y_text + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def compute_map_approx(all_detections):
    """
    Approximate mAP without ground truth:
    - Treat detections of a target class as positives for that class.
    - All other class detections are negatives.
    - Compute AP per class by sorting all detections by confidence and integrating precision over recall
      only at TP positions. mAP is the mean over classes observed so far.
    all_detections: list of tuples (score: float, class_id: int)
    """
    if not all_detections:
        return 0.0
    # Unique classes encountered
    classes = sorted({cid for _, cid in all_detections})
    aps = []
    # Pre-sort once by score desc
    sorted_all = sorted(all_detections, key=lambda x: x[0], reverse=True)
    for c in classes:
        # Total positives for class c (approximated as number of detections predicted as class c)
        P = sum(1 for s, cid in all_detections if cid == c)
        if P == 0:
            continue
        tp = 0
        fp = 0
        prev_recall = 0.0
        ap = 0.0
        for score, cid in sorted_all:
            if cid == c:
                tp += 1
                precision = tp / (tp + fp)
                recall = tp / P
                ap += precision * (recall - prev_recall)
                prev_recall = recall
            else:
                fp += 1
        aps.append(ap)
    if not aps:
        return 0.0
    return float(np.mean(aps))

def prepare_input_tensor(frame_bgr, input_details):
    # Convert frame to model input shape and dtype
    _, in_h, in_w, in_c = input_details['shape']
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h))
    dtype = input_details['dtype']
    if dtype == np.uint8:
        input_data = resized.astype(np.uint8)
    else:
        # Normalize to [0,1] float32
        input_data = (resized.astype(np.float32) / 255.0).astype(dtype)
    # Add batch dimension
    return np.expand_dims(input_data, axis=0)

def extract_detections(interpreter, output_details):
    # Identify boxes, classes, scores, and num detections
    boxes = classes = scores = num = None
    for od in output_details:
        out = interpreter.get_tensor(od['index'])
        shape = out.shape
        if len(shape) == 3 and shape[-1] == 4:
            boxes = out[0]
        elif len(shape) == 2 and shape[-1] > 1 and out.dtype.kind in ('f', 'i'):
            # Could be classes or scores; decide by dtype or value range
            # Heuristic: scores are floats in [0,1]; classes are ints/floats but often >1
            sample = out[0][:min(5, out.shape[-1])]
            if out.dtype.kind == 'f' and np.all((sample >= 0.0) & (sample <= 1.0)):
                scores = out[0]
            else:
                classes = out[0]
        elif len(shape) == 1 and shape[0] == 1:
            num = int(np.squeeze(out))
    # Fallback: if classes or scores misidentified, try alternate assignment
    if classes is None or scores is None:
        # Try to find remaining by scanning again
        for od in output_details:
            out = interpreter.get_tensor(od['index'])
            shape = out.shape
            if len(shape) == 2 and out.shape[-1] > 1:
                if scores is None and out.dtype.kind == 'f' and np.all((out[0] >= 0.0) & (out[0] <= 1.0)):
                    scores = out[0]
                elif classes is None:
                    classes = out[0]
    # If num not provided, infer from boxes or scores
    if num is None:
        if boxes is not None:
            num = boxes.shape[0]
        elif scores is not None:
            num = scores.shape[0]
        else:
            num = 0
    # Trim arrays to num
    if boxes is not None:
        boxes = boxes[:num]
    if classes is not None:
        classes = classes[:num]
    if scores is not None:
        scores = scores[:num]
    return boxes, classes, scores, num

# =========================
# Main Pipeline
# =========================
def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open input video at {input_path}")
        return

    # Prepare video writer with same size as input
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # reasonable default
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, float(fps), (in_w, in_h))
    if not out_writer.isOpened():
        print(f"Error: Cannot open output video for writing at {output_path}")
        cap.release()
        return

    # Accumulate detections for approximate mAP
    all_detections = []
    frame_count = 0
    t0_total = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Prepare input tensor
            input_tensor = prepare_input_tensor(frame, input_details)
            interpreter.set_tensor(input_details['index'], input_tensor)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            inf_ms = (time.time() - t0) * 1000.0

            # Extract detections
            boxes, classes, scores, num = extract_detections(interpreter, output_details)
            if boxes is None or classes is None or scores is None:
                # Nothing to draw
                cv2.putText(frame, "No detections", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                out_writer.write(frame)
                continue

            # Draw detections above threshold and accumulate for mAP
            kept = 0
            for i in range(int(num)):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                cls_raw = classes[i]
                label = class_id_to_label(cls_raw, labels)
                color = get_color_for_class(int(cls_raw))
                draw_labelled_box(frame, boxes[i], label, score, color)
                kept += 1
                # Accumulate detection (score, class_id as int for stable mapping)
                all_detections.append((score, int(cls_raw)))

            # Compute running approximate mAP
            running_map = compute_map_approx(all_detections)
            # Overlay info
            info_text = f"Detections: {kept} | mAP~: {running_map*100:.2f}% | Inference: {inf_ms:.1f} ms"
            cv2.putText(frame, info_text, (10, max(30, in_h - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)

            # Write frame
            out_writer.write(frame)

    finally:
        cap.release()
        out_writer.release()

    total_time = time.time() - t0_total
    final_map = compute_map_approx(all_detections)
    print(f"Processed {frame_count} frames in {total_time:.2f}s ({(frame_count/total_time) if total_time>0 else 0:.2f} FPS).")
    print(f"Approximate mAP over video: {final_map*100:.2f}%")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()