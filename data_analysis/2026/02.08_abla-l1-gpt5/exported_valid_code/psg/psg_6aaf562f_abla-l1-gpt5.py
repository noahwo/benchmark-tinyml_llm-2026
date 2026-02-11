import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5


def ensure_dir_for_file(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_labels(path):
    # Supports both simple one-label-per-line and TFOD 'pbtxt-like' formats (basic).
    labels = []
    id_to_name = {}

    if not os.path.exists(path):
        return labels

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Heuristic: if file contains "item {" assume TFOD-like mapping; else one-per-line.
    if "item" in content and "{" in content and "}" in content and "id:" in content and "name:" in content:
        # Basic parser: find "id:" and "name:" pairs
        current_id = None
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("id:"):
                try:
                    current_id = int(line.split("id:")[1].strip().strip("'\""))
                except Exception:
                    current_id = None
            elif line.startswith("name:"):
                name = line.split("name:")[1].strip().strip("'\"")
                if current_id is not None:
                    id_to_name[current_id] = name
                    current_id = None
        # Convert to list if IDs are contiguous starting at 1 or 0
        if id_to_name:
            max_id = max(id_to_name.keys())
            min_id = min(id_to_name.keys())
            contiguous = sorted(id_to_name.keys()) == list(range(min_id, max_id + 1))
            if contiguous:
                labels = [id_to_name[i] for i in range(min_id, max_id + 1)]
            else:
                # Fallback to sparse mapping stored as list with gaps filled by placeholder
                labels = []
                for i in range(0, max_id + 1):
                    labels.append(id_to_name.get(i, ""))
    else:
        # One label per line
        labels = [line.strip() for line in content.splitlines() if line.strip()]

    return labels


def resolve_label(labels, class_id):
    idx = int(class_id)
    if not labels:
        return f"id:{idx}"
    # If the first label is a background/placeholder, shift index by +1.
    first = labels[0].strip().lower()
    has_background = first in ("???", "background", "bg", "none")
    if has_background and (idx + 1) < len(labels):
        return labels[idx + 1]
    # Otherwise use direct index if in bounds.
    if 0 <= idx < len(labels):
        return labels[idx]
    # Fallback
    return f"id:{idx}"


def get_output_tensors(interpreter):
    # Map outputs by name where possible; fallback to standard ordering.
    output_details = interpreter.get_output_details()

    boxes = scores = classes = num = None
    for od in output_details:
        name = od.get("name", "").lower()
        if "box" in name:
            boxes = interpreter.get_tensor(od["index"])
        elif "score" in name:
            scores = interpreter.get_tensor(od["index"])
        elif "class" in name:
            classes = interpreter.get_tensor(od["index"])
        elif "num" in name or "count" in name:
            num = interpreter.get_tensor(od["index"])

    # Fallback to common SSD ordering if names were not informative
    if boxes is None or scores is None or classes is None:
        try:
            boxes = interpreter.get_tensor(output_details[0]["index"])
            classes = interpreter.get_tensor(output_details[1]["index"])
            scores = interpreter.get_tensor(output_details[2]["index"])
            num = interpreter.get_tensor(output_details[3]["index"]) if len(output_details) > 3 else None
        except Exception:
            raise RuntimeError("Unable to read model outputs. Check model compatibility.")

    # Squeeze batch dimension if present
    boxes = boxes[0] if boxes.ndim == 3 else boxes
    scores = scores[0] if scores.ndim == 2 else scores
    classes = classes[0] if classes.ndim == 2 else classes
    if num is not None and isinstance(num, np.ndarray):
        num = int(np.squeeze(num).astype(np.int32))
    else:
        # If num not provided, infer from scores length
        num = len(scores)

    return boxes, classes, scores, num


def draw_detections(frame, boxes, classes, scores, num, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        cls_name = resolve_label(labels, classes[i])
        caption = f"{cls_name}: {score:.2f}"

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 4)),
                      (x1 + tw + 4, y1), color, thickness=-1)
        # Draw label text
        cv2.putText(frame, caption, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    # 1) SETUP
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter (AI Edge Lite)
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    if not input_details:
        raise RuntimeError("Interpreter has no input details.")
    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    if len(input_shape) != 4 or input_shape[-1] != 3:
        raise RuntimeError(f"Unexpected model input shape: {input_shape}")
    _, in_h, in_w, in_c = input_shape

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    print("Starting inference...")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH}")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Threshold: {CONFIDENCE_THRESHOLD}")

    # 2) PREPROCESSING + 3) INFERENCE + 4) OUTPUT HANDLING
    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

            if input_dtype == np.float32:
                input_tensor = resized.astype(np.float32) / 255.0
            else:
                input_tensor = resized.astype(input_dtype)

            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Set input tensor
            interpreter.set_tensor(input_index, input_tensor)

            # Inference
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer = time.time() - t_infer_start

            # Outputs
            boxes, classes, scores, num = get_output_tensors(interpreter)

            # Draw detections on original BGR frame
            draw_detections(frame_bgr, boxes, classes, scores, num, labels, CONFIDENCE_THRESHOLD)

            # Overlay inference time and FPS
            inst_fps = 1.0 / t_infer if t_infer > 0 else 0.0
            info = f"Inf: {t_infer*1000:.1f} ms | FPS: {inst_fps:.1f}"
            cv2.putText(frame_bgr, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame_bgr)

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    overall_fps = frame_count / total_time if total_time > 0 else 0.0
    print(f"Processed {frame_count} frames in {total_time:.2f}s ({overall_fps:.2f} FPS).")
    print("Done.")


if __name__ == "__main__":
    main()