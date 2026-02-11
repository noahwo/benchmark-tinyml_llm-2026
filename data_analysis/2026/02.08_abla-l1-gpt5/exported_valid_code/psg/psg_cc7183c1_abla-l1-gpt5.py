import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# =========================
# Utilities
# =========================
def load_labels(label_path):
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    # Determine if label map uses a background/??? at index 0
    label_offset = 1 if labels and labels[0].strip().lower() in ("???", "background") else 0
    return labels, label_offset

def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def parse_outputs(interpreter, output_details):
    # Try to map outputs by name first
    boxes = scores = classes = count = None
    for od in output_details:
        name = str(od.get("name", "")).lower()
        data = interpreter.get_tensor(od["index"])
        if "box" in name:
            boxes = data
        elif "score" in name:
            scores = data
        elif "class" in name:
            classes = data
        elif "num" in name or "count" in name:
            count = data

    # Fallback to heuristic mapping if needed
    if boxes is None or scores is None or classes is None:
        outs = [interpreter.get_tensor(od["index"]) for od in output_details]
        for arr in outs:
            arr_squeezed = np.squeeze(arr)
            if arr.ndim >= 3 and arr.shape[-1] == 4:
                boxes = arr
            elif arr.size == 1:
                count = arr
        # Distinguish classes vs scores among remaining arrays
        remaining = [np.squeeze(interpreter.get_tensor(od["index"])) for od in output_details]
        # Remove already identified arrays
        remaining = [r for r in remaining if not (r.ndim >= 2 and r.shape[-1] == 4) and r.size != 1]
        if len(remaining) >= 2:
            # One should be scores in [0,1], the other classes (integers as floats)
            a, b = remaining[0], remaining[1]
            def is_scores(x):
                return x.dtype.kind == 'f' and np.nanmax(x) <= 1.0001 and np.nanmin(x) >= 0.0
            if is_scores(a) and not is_scores(b):
                scores, classes = a, b
            elif is_scores(b) and not is_scores(a):
                scores, classes = b, a
            else:
                # Fallback: choose by variance range
                scores = a
                classes = b

    # Squeeze outputs to expected shapes
    if boxes is not None:
        boxes = np.squeeze(boxes)
    if scores is not None:
        scores = np.squeeze(scores).astype(np.float32)
    if classes is not None:
        classes = np.squeeze(classes).astype(np.int32)
    if count is not None:
        try:
            count = int(np.squeeze(count))
        except Exception:
            count = None

    # Determine number of detections
    if count is None:
        if scores is not None:
            count = scores.shape[0]
        elif boxes is not None:
            count = boxes.shape[0]
        else:
            count = 0

    return boxes, scores, classes, count

# =========================
# Main application
# =========================
def main():
    # 1. Setup: Load labels and initialize interpreter
    labels, label_offset = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input tensor info
    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    # Expected shape [1, height, width, channels]
    model_height = int(input_shape[1])
    model_width = int(input_shape[2])
    input_dtype = input_details[0]["dtype"]

    # 2. Preprocessing: Initialize video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {INPUT_PATH}")
        return

    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 1e-2:
        fps = 30.0  # default fallback

    ensure_dir_exists(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (src_width, src_height))
    if not writer.isOpened():
        print(f"Error: Could not open output video for write: {OUTPUT_PATH}")
        cap.release()
        return

    # Timing for FPS
    prev_time = time.perf_counter()

    # 3. Inference Loop
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Resize and convert BGR to RGB
        resized_bgr = cv2.resize(frame_bgr, (model_width, model_height))
        input_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        # Prepare input tensor
        if input_dtype == np.float32:
            input_tensor = (input_rgb.astype(np.float32) / 255.0)
        else:
            input_tensor = input_rgb.astype(np.uint8)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Set tensor and run inference
        interpreter.set_tensor(input_index, input_tensor)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        # 4. Output handling: parse detections and draw
        boxes, scores, classes, count = parse_outputs(interpreter, output_details)

        # Draw detections
        if boxes is not None and scores is not None and classes is not None and count > 0:
            for i in range(count):
                score = float(scores[i])
                if score < CONFIDENCE_THRESHOLD:
                    continue

                # Boxes are in normalized coordinates [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = boxes[i]
                left = int(max(0, xmin * src_width))
                top = int(max(0, ymin * src_height))
                right = int(min(src_width - 1, xmax * src_width))
                bottom = int(min(src_height - 1, ymax * src_height))

                cls_id = int(classes[i]) + label_offset
                if 0 <= cls_id < len(labels):
                    cls_name = labels[cls_id]
                else:
                    cls_name = f"id_{int(classes[i])}"

                color = (0, 255, 0)
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

                label_text = f"{cls_name}: {score:.2f}"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = left
                text_y = max(0, top - 8)
                cv2.rectangle(frame_bgr, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), (0, 0, 0), -1)
                cv2.putText(frame_bgr, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Compute and draw FPS and inference time
        now = time.perf_counter()
        fps_inst = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        info_text = f"Inference: {inference_ms:.1f} ms | FPS: {fps_inst:.1f}"
        cv2.putText(frame_bgr, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)

        # Write frame to output
        writer.write(frame_bgr)

    # Cleanup
    cap.release()
    writer.release()

if __name__ == "__main__":
    main()