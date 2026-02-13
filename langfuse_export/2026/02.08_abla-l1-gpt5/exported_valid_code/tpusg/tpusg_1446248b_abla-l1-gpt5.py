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

EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(label_path):
    labels = {}
    try:
        with open(label_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Try "index: label"
                if ":" in line:
                    left, right = line.split(":", 1)
                    left = left.strip()
                    right = right.strip()
                    try:
                        idx = int(left)
                        labels[idx] = right
                        continue
                    except ValueError:
                        pass
                # Try "index label"
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                    continue
                # Fallback: use line order
                labels[i] = line
    except Exception as e:
        print(f"Warning: Failed to load labels from {label_path}: {e}")
    return labels


def make_interpreter(model_path):
    # Try to create interpreter with EdgeTPU delegate
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)]
        )
        print("EdgeTPU delegate loaded successfully.")
        return interpreter
    except Exception as e:
        print(f"Warning: Failed to load EdgeTPU delegate ({e}). Falling back to CPU.")
        return Interpreter(model_path=model_path)


def get_output_tensors(interpreter):
    # Parse common TFLite detection postprocess outputs
    output_details = interpreter.get_output_details()
    outputs_by_name = {d['name'].lower(): d for d in output_details}
    boxes = classes = scores = count = None

    # Prefer by name when available
    for d in output_details:
        name = d['name'].lower()
        if 'box' in name:
            boxes = interpreter.get_tensor(d['index'])
        elif 'score' in name:
            scores = interpreter.get_tensor(d['index'])
        elif 'class' in name:
            classes = interpreter.get_tensor(d['index'])
        elif 'num' in name:
            count = interpreter.get_tensor(d['index'])

    # Fallback by shape heuristics
    if boxes is None or scores is None or classes is None or count is None:
        tensors = [interpreter.get_tensor(d['index']) for d in output_details]
        # Try to infer
        for t in tensors:
            if boxes is None and len(t.shape) == 3 and t.shape[-1] == 4:
                boxes = t
        for t in tensors:
            if count is None and t.shape == (1,) and np.issubdtype(t.dtype, np.integer):
                count = t
        # Scores and classes (both [1, num])
        candidates = [t for t in tensors if len(t.shape) == 2 and t.shape[0] == 1]
        # Heuristic: scores in [0,1]
        for t in candidates:
            if scores is None and np.issubdtype(t.dtype, np.floating):
                # Check if values are within [0,1] mostly
                vals = t[0]
                if np.all((vals >= -1e-6) & (vals <= 1.0 + 1e-6)):
                    scores = t
                    break
        # Classes: remaining candidate
        if classes is None:
            for t in candidates:
                if t is not scores:
                    classes = t
                    break

    return boxes, classes, scores, count


def set_input_tensor(interpreter, frame_bgr):
    # Prepare input tensor with resizing and color conversion
    input_details = interpreter.get_input_details()[0]
    height, width = input_details['shape'][1], input_details['shape'][2]
    dtype = input_details['dtype']

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    input_data = resized
    if dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        # Generic cast
        input_data = input_data.astype(dtype)

    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details['index'], input_data)


def run_inference(interpreter, frame_bgr, orig_width, orig_height, conf_threshold=0.5):
    set_input_tensor(interpreter, frame_bgr)

    t0 = time.time()
    interpreter.invoke()
    infer_ms = (time.time() - t0) * 1000.0

    boxes, classes, scores, count = get_output_tensors(interpreter)
    detections = []

    if boxes is None or classes is None or scores is None or count is None:
        # Unable to parse model outputs
        return detections, infer_ms

    num = int(count[0]) if np.ndim(count) > 0 else int(count)
    b = boxes[0]
    c = classes[0]
    s = scores[0]

    for i in range(min(num, b.shape[0], c.shape[0], s.shape[0])):
        score = float(s[i])
        if score < conf_threshold:
            continue
        class_id = int(c[i])
        # boxes are [ymin, xmin, ymax, xmax] in normalized coordinates
        ymin, xmin, ymax, xmax = b[i]
        x1 = int(max(0, xmin) * orig_width)
        y1 = int(max(0, ymin) * orig_height)
        x2 = int(min(1.0, xmax) * orig_width)
        y2 = int(min(1.0, ymax) * orig_height)
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": class_id
        })

    return detections, infer_ms


def draw_detections(frame_bgr, detections, labels):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        class_id = det["class_id"]
        label = labels.get(class_id, str(class_id))
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        caption = f"{label}: {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_bgr, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def overlay_metrics(frame_bgr, fps, map_value, conf_threshold):
    h, w = frame_bgr.shape[:2]
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    map_text = "N/A" if map_value is None else f"{map_value:.3f}"
    lines = [
        f"FPS: {fps:.1f}",
        f"mAP: {map_text}",
        f"Threshold: {conf_threshold:.2f}"
    ]

    # Draw semi-transparent box
    padding = 8
    line_height = 22
    box_w = max([cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines]) + padding * 2
    box_h = line_height * len(lines) + padding
    x0, y0 = 10, 10
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    # Put text
    for i, line in enumerate(lines):
        y = y0 + padding + (i + 1) * (line_height - 5)
        cv2.putText(frame_bgr, line, (x0 + padding, y), font, scale, text_color, thickness, cv2.LINE_AA)


def main():
    # 1. Setup and initialization
    labels = load_labels(LABEL_PATH)

    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {INPUT_PATH}")
        return

    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # Default fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_width, in_height))
    if not writer.isOpened():
        print(f"Error: Could not open output writer: {OUTPUT_PATH}")
        cap.release()
        return

    # Variables for performance and mAP placeholder (no GT available)
    frame_times = []
    map_value = None  # mAP cannot be computed without ground truth; will display N/A.

    # 2-3-4. Process frames: preprocess, inference, draw, metrics, save
    frame_index = 0
    start_time_overall = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            t0 = time.time()
            detections, infer_ms = run_inference(interpreter, frame, in_width, in_height, CONFIDENCE_THRESHOLD)
            draw_detections(frame, detections, labels)

            # Compute FPS using moving average of frame processing time
            elapsed_ms = (time.time() - t0) * 1000.0
            frame_times.append(elapsed_ms)
            if len(frame_times) > 60:
                frame_times.pop(0)
            avg_ms = np.mean(frame_times) if frame_times else elapsed_ms
            current_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

            # Overlay metrics including mAP placeholder
            overlay_metrics(frame, current_fps, map_value, CONFIDENCE_THRESHOLD)

            # Write frame
            writer.write(frame)

            # Optional: progress print every 50 frames
            if frame_index % 50 == 0:
                print(f"Processed {frame_index}/{total_frames if total_frames>0 else '?'} frames. Inference: {infer_ms:.2f} ms, FPS: {current_fps:.1f}")
    finally:
        cap.release()
        writer.release()

    total_time = time.time() - start_time_overall
    print(f"Done. Processed {frame_index} frames in {total_time:.2f}s ({(frame_index/total_time) if total_time>0 else 0:.1f} FPS avg).")
    print(f"Output saved to: {OUTPUT_PATH}")
    if map_value is None:
        print("Note: mAP could not be computed because no ground-truth annotations were provided.")


if __name__ == "__main__":
    main()