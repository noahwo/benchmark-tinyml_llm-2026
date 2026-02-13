import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Attempt to parse "index label" format; fallback to enumerated labels
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[idx] = line
    return labels


def set_input_tensor(interpreter, image_rgb):
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    input_dtype = input_details['dtype']
    input_shape = input_details['shape']  # [1, height, width, 3]
    height, width = input_shape[1], input_shape[2]

    resized = cv2.resize(image_rgb, (width, height))
    if input_dtype == np.float32:
        # Normalize to [0,1]
        input_data = resized.astype(np.float32) / 255.0
    else:
        # Assume quantized uint8
        input_data = resized.astype(np.uint8)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_index, input_data)


def get_output_tensors(interpreter):
    """Retrieve detection outputs in a model-agnostic way, dequantize if required."""
    output_details = interpreter.get_output_details()

    def dequantize(output_detail, data):
        if np.issubdtype(output_detail['dtype'], np.floating):
            return data
        scale, zero_point = output_detail['quantization']
        if scale == 0:
            return data.astype(np.float32)
        return scale * (data.astype(np.float32) - zero_point)

    boxes = None
    classes = None
    scores = None
    count = None

    # Try to identify outputs by shape semantics
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        dq = dequantize(od, data)

        # Boxes: shape [1, N, 4]
        if len(dq.shape) == 3 and dq.shape[0] == 1 and dq.shape[2] == 4:
            boxes = dq[0]
        # Classes: shape [1, N]
        elif len(dq.shape) == 2 and dq.shape[0] == 1 and np.issubdtype(dq.dtype, np.floating):
            # Scores and classes could both be [1, N] float; try to distinguish by value range
            # Heuristic: classes are near small integers; scores in [0,1]
            if np.all((dq >= -1) & (dq <= len(dq[0]) + 10)) and np.max(dq) > 1.0:
                classes = dq[0]
            else:
                # Could be scores if within [0,1]
                if np.max(dq) <= 1.0001:
                    scores = dq[0]
                else:
                    # If ambiguous, will resolve later if one is missing
                    pass
        # Count: shape [1] or scalar
        elif dq.size == 1:
            count = int(np.squeeze(dq))

    # If ambiguity remains between classes and scores, try to swap if needed
    if classes is not None and scores is not None:
        pass
    elif classes is None or scores is None:
        # Try a different interpretation based on typical TFLite ordering
        floats = [interpreter.get_tensor(od['index']).astype(np.float32) for od in output_details]
        # Look for arrays in [0,1] as scores
        cand_scores = [f for f in floats if f.ndim == 2 and f.shape[0] == 1 and np.max(f) <= 1.0001]
        if scores is None and len(cand_scores) > 0:
            scores = cand_scores[0][0]
        # Look for arrays with larger integers as classes
        cand_classes = [f for f in floats if f.ndim == 2 and f.shape[0] == 1 and np.max(f) > 1.0]
        if classes is None and len(cand_classes) > 0:
            classes = cand_classes[0][0]

    # Fallbacks
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((len(boxes),), dtype=np.float32)
    if scores is None:
        scores = np.zeros((len(boxes),), dtype=np.float32)
    if count is None:
        count = len(boxes)

    # Ensure correct sizes
    n = min(int(count), boxes.shape[0], classes.shape[0], scores.shape[0])
    return boxes[:n], classes[:n], scores[:n], n


def draw_detections(frame, detections, labels, map_value=None):
    h, w = frame.shape[:2]
    for det in detections:
        ymin, xmin, ymax, xmax = det['bbox']  # normalized
        score = det['score']
        cls_id = det['class_id']
        x1 = max(0, int(xmin * w))
        y1 = max(0, int(ymin * h))
        x2 = min(w - 1, int(xmax * w))
        y2 = min(h - 1, int(ymax * h))

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = labels.get(cls_id, f"id:{cls_id}")
        text = f"{label} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if map_value is not None:
        map_text = f"mAP: {map_value:.3f}"
        (tw, th), bl = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (5, 5), (5 + tw + 10, 5 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, map_text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter with EdgeTPU
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate(EDGETPU_LIB)]
    )
    interpreter.allocate_tensors()

    # Get input size info
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details['shape']
    in_height, in_width = input_shape[1], input_shape[2]

    # Video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 1e-2:
        input_fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, input_fps, (frame_w, frame_h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # Stats for "mAP" approximation (proxy due to lack of ground truth)
    # We'll treat AP per-class as the mean confidence of detections above threshold, then mAP is mean over classes observed.
    per_class_confidences = {}

    frame_count = 0
    t0_all = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # Preprocessing: BGR -> RGB and set tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        set_input_tensor(interpreter, frame_rgb)

        # Inference
        t0 = time.time()
        interpreter.invoke()
        infer_time_ms = (time.time() - t0) * 1000.0

        # Outputs
        boxes, classes, scores, count = get_output_tensors(interpreter)

        # Collect detections above threshold
        detections = []
        for i in range(count):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(classes[i])
            ymin, xmin, ymax, xmax = boxes[i]  # normalized [ymin, xmin, ymax, xmax]
            ymin = float(max(0.0, min(1.0, ymin)))
            xmin = float(max(0.0, min(1.0, xmin)))
            ymax = float(max(0.0, min(1.0, ymax)))
            xmax = float(max(0.0, min(1.0, xmax)))
            detections.append({
                'bbox': (ymin, xmin, ymax, xmax),
                'score': score,
                'class_id': cls_id
            })
            # Update proxy AP stats
            if cls_id not in per_class_confidences:
                per_class_confidences[cls_id] = []
            per_class_confidences[cls_id].append(score)

        # Compute proxy mAP
        if per_class_confidences:
            ap_values = [np.mean(per_class_confidences[c]) for c in per_class_confidences]
            map_value = float(np.mean(ap_values))
        else:
            map_value = 0.0

        # Draw and write
        draw_detections(frame_bgr, detections, labels, map_value=map_value)
        # Optionally draw FPS/inference time
        perf_text = f"Infer: {infer_time_ms:.1f} ms"
        cv2.putText(frame_bgr, perf_text, (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        out.write(frame_bgr)

    total_time = time.time() - t0_all
    cap.release()
    out.release()

    # Final report
    if per_class_confidences:
        ap_values = {c: float(np.mean(per_class_confidences[c])) for c in per_class_confidences}
        final_map = float(np.mean(list(ap_values.values())))
    else:
        ap_values = {}
        final_map = 0.0

    print(f"Processed {frame_count} frames in {total_time:.2f}s.")
    print(f"Approx. mAP (no ground-truth; mean of per-class mean confidences): {final_map:.4f}")
    if ap_values:
        print("Per-class AP (approx):")
        for c, ap in ap_values.items():
            name = labels.get(c, f"id:{c}")
            print(f"  {name} (id {c}): {ap:.4f}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()