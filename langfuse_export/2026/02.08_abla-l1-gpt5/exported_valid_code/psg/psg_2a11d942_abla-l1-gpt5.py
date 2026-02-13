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
    try:
        with open(path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    except Exception as e:
        print(f"Failed to load labels from {path}: {e}")
    return labels

def ensure_dir_for_file(file_path):
    out_dir = os.path.dirname(file_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

def initialize_interpreter(model_path):
    # 1) setup: initialize interpreter and get I/O details
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Assume single input tensor
    input_shape = input_details[0]['shape']
    input_height, input_width = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    return interpreter, input_details, output_details, (input_height, input_width), input_dtype

def preprocess_frame(frame_bgr, input_size, input_dtype):
    # 2) preprocessing: BGR -> RGB, resize, convert dtype, normalize if needed
    ih, iw = input_size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih))
    if input_dtype == np.float32:
        input_tensor = resized.astype(np.float32) / 255.0
    else:
        input_tensor = resized.astype(np.uint8)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def run_inference(interpreter, input_details, output_details, input_tensor):
    # 3) inference: set input, invoke, get outputs
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Typical TFLite SSD outputs order: boxes, classes, scores, num_detections
    # boxes: [1, N, 4], classes: [1, N], scores: [1, N], num: [1]
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    # Squeeze batch dimension
    boxes = np.squeeze(boxes, axis=0)
    classes = np.squeeze(classes, axis=0)
    scores = np.squeeze(scores, axis=0)
    # num can be float in some models
    num_detections = int(np.squeeze(num))
    return boxes, classes, scores, num_detections

def draw_detections(frame_bgr, boxes, classes, scores, num_detections, labels, threshold=0.5):
    h, w = frame_bgr.shape[:2]
    for i in range(num_detections):
        score = float(scores[i])
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        # Convert normalized coordinates to absolute pixel values
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        class_id = int(classes[i]) if i < len(classes) else -1
        label = labels[class_id] if 0 <= class_id < len(labels) else f"id:{class_id}"

        # Draw bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label and score
        caption = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame_bgr, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter
    interpreter, input_details, output_details, input_size, input_dtype = initialize_interpreter(MODEL_PATH)

    # Prepare video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Failed to open input video: {INPUT_PATH}")
        return

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback

    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        print(f"Failed to open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            input_tensor = preprocess_frame(frame, input_size, input_dtype)

            # Inference
            boxes, classes, scores, num_detections = run_inference(interpreter, input_details, output_details, input_tensor)

            # Output handling: draw detections
            draw_detections(frame, boxes, classes, scores, num_detections, labels, threshold=CONFIDENCE_THRESHOLD)

            # Write frame to output
            writer.write(frame)

            frame_count += 1
    finally:
        cap.release()
        writer.release()

    t1 = time.time()
    elapsed = t1 - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s. Average FPS: {avg_fps:.2f}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()