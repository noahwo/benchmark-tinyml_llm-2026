import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(path):
    labels = []
    if not os.path.isfile(path):
        print(f"Label file not found: {path}")
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try to parse formats like "0: person" or "0 person"
            label = line
            if ":" in line:
                parts = line.split(":", 1)
                left = parts[0].strip()
                right = parts[1].strip()
                if left.isdigit():
                    label = right
            else:
                parts = line.split()
                if len(parts) > 1 and parts[0].isdigit():
                    label = " ".join(parts[1:])
            labels.append(label)
    return labels

def make_output_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def preprocess_frame(frame_bgr, input_size, input_dtype, input_quantization):
    # Convert BGR to RGB and resize
    h_in, w_in = input_size
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (w_in, h_in))
    if input_dtype == np.uint8:
        # For quantized models, pass uint8 data directly
        input_tensor = resized.astype(np.uint8)
    else:
        # For float models, normalize to [0,1]
        input_tensor = resized.astype(np.float32) / 255.0
    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def parse_outputs(interpreter):
    output_details = interpreter.get_output_details()

    boxes = None
    classes = None
    scores = None
    count = None

    # First try by output tensor names
    for od in output_details:
        name = od.get('name', '').lower()
        data = interpreter.get_tensor(od['index'])
        if 'box' in name:
            boxes = data[0]
        elif 'score' in name:
            scores = data[0]
        elif 'class' in name:
            classes = data[0].astype(np.int32)
        elif 'num' in name:
            # Some models return float count
            count_val = data.flatten()[0]
            try:
                count = int(count_val)
            except Exception:
                count = int(round(float(count_val)))

    # Fallback by shapes/values if names were not descriptive
    if boxes is None or classes is None or scores is None or count is None:
        # Re-fetch so we don't lose anything if we partially parsed above
        tensors = [interpreter.get_tensor(od['index']) for od in output_details]
        # Identify boxes (shape [1, N, 4])
        for t in tensors:
            if isinstance(t, np.ndarray) and t.ndim == 3 and t.shape[0] == 1 and t.shape[2] == 4:
                boxes = t[0]
                break
        # Identify scores and classes among [1, N]
        candidates = [t[0] for t in tensors if isinstance(t, np.ndarray) and t.ndim == 2 and t.shape[0] == 1]
        if len(candidates) >= 2:
            # Heuristic: scores in [0,1], classes >= 0 and typically >1
            c1, c2 = candidates[0], candidates[1]
            def is_scores(arr):
                return float(np.nanmax(arr)) <= 1.0001
            if is_scores(c1) and not is_scores(c2):
                scores, classes = c1, c2.astype(np.int32)
            elif is_scores(c2) and not is_scores(c1):
                scores, classes = c2, c1.astype(np.int32)
            else:
                # If ambiguous, assume first is scores
                scores, classes = c1, c2.astype(np.int32)
        # Identify count [1] or [1,1]
        for t in tensors:
            if isinstance(t, np.ndarray) and t.size == 1:
                try:
                    count = int(round(float(t.flatten()[0])))
                    break
                except Exception:
                    pass

    # Final sanity defaults if something missing
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.int32)
    if count is None:
        count = min(len(scores), len(boxes))

    # Ensure consistent length
    n = min(count, len(scores), len(boxes), len(classes))
    return boxes[:n], classes[:n], scores[:n], n

def draw_detections(frame_bgr, boxes, classes, scores, labels, threshold):
    h, w = frame_bgr.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        # Convert normalized coords to absolute pixel coordinates if needed
        if 0.0 <= xmin <= 1.0 and 0.0 <= xmax <= 1.0 and 0.0 <= ymin <= 1.0 and 0.0 <= ymax <= 1.0:
            x1 = int(max(0, xmin * w))
            y1 = int(max(0, ymin * h))
            x2 = int(min(w - 1, xmax * w))
            y2 = int(min(h - 1, ymax * h))
        else:
            # Already absolute (rare)
            x1 = int(max(0, xmin))
            y1 = int(max(0, ymin))
            x2 = int(min(w - 1, xmax))
            y2 = int(min(h - 1, ymax))

        cls_id = int(classes[i]) if i < len(classes) else -1
        label_text = str(cls_id)
        if 0 <= cls_id < len(labels):
            label_text = labels[cls_id]
        caption = f"{label_text}: {score:.2f}"

        # Draw bounding box
        color = (0, 255, 0)  # green
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Draw label background and text
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - baseline - 4)),
                      (x1 + tw + 4, y1), color, thickness=-1)
        cv2.putText(frame_bgr, caption, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    # Step 1: Setup (load labels, initialize interpreter)
    labels = load_labels(LABEL_PATH)

    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    if not input_details:
        print("No input details found in the model.")
        return
    input_shape = input_details[0]['shape']
    # Expected shape [1, height, width, 3]
    in_height = int(input_shape[1])
    in_width = int(input_shape[2])
    in_dtype = input_details[0]['dtype']
    in_quant = input_details[0].get('quantization', (0.0, 0))

    # Step 2: Preprocessing (video setup and frame preparation)
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Failed to open input video: {INPUT_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    make_output_dir(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"Failed to open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Step 3 and 4: Inference loop and output handling (annotate and save)
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess frame
            input_tensor = preprocess_frame(frame, (in_height, in_width), in_dtype, in_quant)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            # Parse outputs
            boxes, classes, scores, count = parse_outputs(interpreter)

            # Draw detections
            draw_detections(frame, boxes, classes, scores, labels, CONFIDENCE_THRESHOLD)

            # Write frame to output
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    t1 = time.time()
    elapsed = t1 - t0
    if elapsed > 0:
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} FPS).")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()