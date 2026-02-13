import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5


def load_labels(label_path):
    """
    Loads labels from a file.
    Supports either:
      - one label per line (index inferred by line number), or
      - "index label" per line (index parsed explicitly).
    Returns a list where index corresponds to class id.
    """
    labels_map = {}
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels_map[int(parts[0])] = parts[1].strip()
                else:
                    labels_map[i] = line
    except Exception as e:
        print(f"Warning: Failed to read labels from {label_path}: {e}")
        return []

    if not labels_map:
        return []

    max_id = max(labels_map.keys())
    labels = [""] * (max_id + 1)
    for k, v in labels_map.items():
        if k < 0:
            continue
        if k >= len(labels):
            labels.extend([""] * (k - len(labels) + 1))
        labels[k] = v
    return labels


def preprocess_frame(frame, input_size, input_dtype):
    """
    Preprocess frame for model input:
    - Resize to model input size
    - Convert BGR to RGB
    - Convert dtype and scale if needed
    - Add batch dimension
    """
    h, w = input_size
    resized = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        tensor = (rgb.astype(np.float32) / 255.0)[None, ...]
    else:
        tensor = rgb.astype(np.uint8)[None, ...]
    return tensor


def draw_detections(frame, boxes, classes, scores, labels, threshold):
    """
    Draws detection boxes and labels on the frame.
    boxes: (N, 4) with values in [0,1] as [ymin, xmin, ymax, xmax]
    classes: (N,) int
    scores: (N,) float
    """
    height, width = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue

        cls_id = int(classes[i]) if classes is not None else -1
        label = labels[cls_id] if labels and 0 <= cls_id < len(labels) and labels[cls_id] else f"id:{cls_id}"
        caption = f"{label} {score:.2f}"

        box = boxes[i]
        ymin = max(0, min(height - 1, int(box[0] * height)))
        xmin = max(0, min(width - 1, int(box[1] * width)))
        ymax = max(0, min(height - 1, int(box[2] * height)))
        xmax = max(0, min(width - 1, int(box[3] * width)))

        color = (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, max(0, ymin - text_h - baseline - 4)),
                      (xmin + text_w + 4, ymin), color, thickness=-1)
        cv2.putText(frame, caption, (xmin + 2, ymin - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)


def main():
    # 1) SETUP
    # Create output directory
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_h, in_w = input_details[0]['shape'][1:3]
    in_dtype = input_details[0]['dtype']
    input_index = input_details[0]['index']

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video at {INPUT_PATH}")
        return

    # Video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing at {OUTPUT_PATH}")
        cap.release()
        return

    print("Starting inference...")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH}")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # 2) PREPROCESSING + 3) INFERENCE + 4) OUTPUT HANDLING
    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Preprocess frame for the model
        input_tensor = preprocess_frame(frame, (in_h, in_w), in_dtype)

        # Inference
        interpreter.set_tensor(input_index, input_tensor)
        t_infer_start = time.time()
        interpreter.invoke()
        t_infer = (time.time() - t_infer_start) * 1000.0  # ms

        # Gather outputs robustly
        boxes = None
        classes = None
        scores = None
        num_det = None

        for od in output_details:
            out = interpreter.get_tensor(od['index'])
            out = np.squeeze(out)

            # Identify outputs based on shape/value ranges
            if out.ndim == 2 and out.shape[1] == 4:
                boxes = out
            elif out.ndim == 1 and out.size == 4 and boxes is None:
                # Edge case (rare), not typical for SSD. Keep for completeness.
                boxes = out.reshape(1, 4)
            elif out.ndim == 1 and out.size > 1:
                if np.max(out) <= 1.0 and np.min(out) >= 0.0:
                    scores = out
                else:
                    classes = out.astype(int)
            elif out.ndim == 0 or (out.ndim == 1 and out.size == 1):
                try:
                    num_det = int(round(float(out)))
                except Exception:
                    num_det = None

        # Fallbacks if num_det is not provided
        if scores is not None and num_det is None:
            num_det = scores.shape[0]
        if boxes is not None and num_det is None:
            num_det = boxes.shape[0]
        if classes is not None and num_det is None:
            num_det = classes.shape[0]

        # Slice to num_det
        if boxes is not None:
            boxes = boxes[:num_det]
        if classes is not None:
            classes = classes[:num_det]
        if scores is not None:
            scores = scores[:num_det]

        # Draw results
        if boxes is not None and scores is not None:
            draw_detections(frame, boxes, classes if classes is not None else np.zeros_like(scores, dtype=int),
                            scores, labels, CONFIDENCE_THRESHOLD)

        # Overlay inference time and FPS
        elapsed = time.time() - t0
        avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
        info_text = f"Inf: {t_infer:.1f}ms | Avg FPS: {avg_fps:.1f}"
        cv2.putText(frame, info_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, lineType=cv2.LINE_AA)

        # Write to output
        writer.write(frame)

    cap.release()
    writer.release()
    print("Processing completed.")
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()