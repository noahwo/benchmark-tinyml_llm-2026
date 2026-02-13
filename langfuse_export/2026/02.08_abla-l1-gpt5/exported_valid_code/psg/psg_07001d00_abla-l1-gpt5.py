import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters (as provided)
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5


def load_labels(label_path):
    """
    Loads labels from a file. Supports:
    - Plain text (one label per line; optional first '???' placeholder).
    - Simple 'item { id: X name: "Y" }' style label maps.
    Returns: dict mapping class_id (int) -> label (str)
    """
    if not os.path.isfile(label_path):
        return {}

    with open(label_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try to parse 'item { id: X name: "Y" }' style
    if "item" in content and "id" in content and "name" in content:
        labels = {}
        id_val, name_val = None, None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("id:"):
                try:
                    id_val = int(line.split("id:")[1].strip())
                except Exception:
                    id_val = None
            if line.startswith("name:"):
                name_val = line.split("name:")[1].strip().strip('"').strip("'")
            if line.startswith("}"):
                if id_val is not None and name_val is not None:
                    labels[id_val] = name_val
                id_val, name_val = None, None
        if labels:
            return labels

    # Fallback: plain text file, one label per line
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return {i: lbl for i, lbl in enumerate(lines)}


def preprocess_frame(frame_bgr, input_size, input_dtype):
    """
    Preprocessing:
    - Convert BGR to RGB
    - Resize to model input size
    - Convert dtype (uint8 or float32 in [0,1])
    - Add batch dimension
    """
    h_in, w_in = input_size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.uint8:
        input_tensor = resized.astype(np.uint8)
    else:
        # Assume float32 model expects [0,1]
        input_tensor = (resized.astype(np.float32) / 255.0).astype(np.float32)

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def draw_detections(frame_bgr, boxes, classes, scores, labels, threshold):
    """
    Draw bounding boxes and labels on the frame for detections above threshold.
    boxes: normalized [ymin, xmin, ymax, xmax]
    """
    h, w = frame_bgr.shape[:2]

    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue

        cls_id = int(classes[i])
        label_text = labels.get(cls_id, str(cls_id))

        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, int(xmin * w))
        y1 = max(0, int(ymin * h))
        x2 = min(w - 1, int(xmax * w))
        y2 = min(h - 1, int(ymax * h))

        color = (37 * (cls_id % 7) + 20, 17 * (cls_id % 17) + 50, 29 * (cls_id % 13) + 80)
        color = tuple(int(c % 255) for c in color)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        caption = f"{label_text}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame_bgr, caption, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame_bgr


def main():
    # Validate input
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(LABEL_PATH):
        print(f"Warning: Label file not found: {LABEL_PATH}. Proceeding without class names.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. Setup: Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input tensor info
    in_index = input_details[0]["index"]
    # Expected shape: [1, height, width, 3]
    in_height = int(input_details[0]["shape"][1])
    in_width = int(input_details[0]["shape"][2])
    in_dtype = input_details[0]["dtype"]

    # Output indices: TFLite SSD models typically return [boxes, classes, scores, num]
    # We'll assume standard order but also verify shapes.
    if len(output_details) < 3:
        raise RuntimeError("Unexpected model outputs. Expected at least 3 outputs for detection.")

    # Heuristic assignment (commonly: 0=boxes, 1=classes, 2=scores, 3=num)
    # Fallback to shape-based selection for boxes.
    boxes_idx = None
    for i, od in enumerate(output_details):
        shp = od["shape"]
        if len(shp) == 3 and shp[-1] == 4:
            boxes_idx = i
            break
    if boxes_idx is None:
        # Default to first
        boxes_idx = 0

    # Identify remaining two likely indices for classes and scores
    remaining_idxs = [i for i in range(len(output_details)) if i != boxes_idx]
    # Prefer to find num_detections (shape [1] or [1,1]) and exclude it
    maybe_num_idx = None
    for i in remaining_idxs:
        shp = output_details[i]["shape"]
        if (len(shp) == 1 and shp[0] == 1) or (len(shp) == 2 and shp[0] == 1 and shp[1] == 1):
            maybe_num_idx = i
            break
    if maybe_num_idx is not None:
        remaining_idxs.remove(maybe_num_idx)

    # Now remaining two should be classes and scores; assume lower index is classes, higher is scores per common pattern
    if len(remaining_idxs) >= 2:
        classes_idx, scores_idx = remaining_idxs[0], remaining_idxs[1]
    elif len(remaining_idxs) == 1:
        # If only one remains, assume it's scores and classes will be inferred as same (some models omit classes)
        classes_idx, scores_idx = remaining_idxs[0], remaining_idxs[0]
    else:
        # Fallback to defaults
        classes_idx, scores_idx = 1, 2

    # Load labels
    labels = load_labels(LABEL_PATH)

    # 2. Video setup (I/O)
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps is None or in_fps <= 0:
        in_fps = 30.0  # fallback

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # Performance tracking
    frame_count = 0
    t_start = time.time()
    last_log = t_start

    # 3. Processing loop: preprocessing -> inference -> output handling
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # 2. Preprocessing
            input_tensor = preprocess_frame(frame_bgr, (in_height, in_width), in_dtype)

            # 3. Inference
            interpreter.set_tensor(in_index, input_tensor)
            interpreter.invoke()

            # 4. Output handling
            boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])
            # Normalize shapes to [N]
            classes = interpreter.get_tensor(output_details[classes_idx]["index"])
            scores = interpreter.get_tensor(output_details[scores_idx]["index"])

            # Remove batch dimension
            if boxes.ndim == 3:
                boxes = boxes[0]
            if classes.ndim == 2:
                classes = classes[0]
            if scores.ndim == 2:
                scores = scores[0]

            # Draw detections
            annotated = draw_detections(
                frame_bgr,
                boxes=boxes,
                classes=classes,
                scores=scores,
                labels=labels,
                threshold=CONFIDENCE_THRESHOLD
            )

            writer.write(annotated)
            frame_count += 1

            # Periodic logging
            now = time.time()
            if now - last_log >= 2.0:
                elapsed = now - t_start
                fps_avg = frame_count / max(elapsed, 1e-6)
                print(f"Processed {frame_count} frames | Avg FPS: {fps_avg:.2f}")
                last_log = now

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 1e-6)
    print(f"Done. Wrote: {OUTPUT_PATH}")
    print(f"Total frames: {frame_count}, Total time: {total_time:.2f}s, Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()