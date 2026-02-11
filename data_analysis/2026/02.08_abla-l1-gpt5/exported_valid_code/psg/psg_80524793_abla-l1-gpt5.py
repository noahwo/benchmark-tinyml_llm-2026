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
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def get_label_offset(labels):
    if not labels:
        return 0
    first = labels[0].strip().lower()
    if first in ('???', 'background'):
        return 1
    return 0

def create_video_writer(output_path, width, height, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Use mp4v codec for .mp4 output, which is commonly available on Raspberry Pi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25.0, (width, height))
    return writer

def preprocess_frame(frame, input_size, input_dtype):
    # Convert BGR to RGB, resize to model input size, and format based on dtype
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size[1], input_size[0]))  # (width, height) vs (height, width)
    if input_dtype == np.uint8:
        input_tensor = np.expand_dims(resized, axis=0)
    else:
        # Assume float32 expects [0,1] normalization
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return input_tensor

def draw_detections(frame, boxes, classes, scores, labels, label_offset, threshold):
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue

        # boxes in format [ymin, xmin, ymax, xmax], normalized 0..1
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        class_id = int(classes[i])
        label_index = class_id + label_offset
        if 0 <= label_index < len(labels):
            label_text = labels[label_index]
        else:
            label_text = f"id:{class_id}"

        caption = f"{label_text}: {score*100:.1f}%"

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Text background for readability
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline)
        cv2.rectangle(frame, (x1, y_text), (x1 + tw + 4, y_text + th + baseline), (0, 0, 0), -1)
        cv2.putText(frame, caption, (x1 + 2, y_text + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def parse_outputs(interpreter, output_details):
    # Initialize containers
    boxes = scores = classes = num = None

    # Try to identify outputs by name first
    for od in output_details:
        name = od.get('name', '')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='ignore')
        data = interpreter.get_tensor(od['index'])
        if name and 'boxes' in name:
            boxes = data[0]
        elif name and 'scores' in name:
            scores = data[0]
        elif name and 'classes' in name:
            classes = data[0]
        elif name and 'num' in name:
            num = int(np.squeeze(data).astype(np.int32))

    # If any remain unidentified, infer by shape/value ranges
    if boxes is None or scores is None or classes is None:
        for od in output_details:
            data = interpreter.get_tensor(od['index'])
            shp = data.shape
            if boxes is None and len(shp) == 3 and shp[-1] == 4:
                boxes = data[0]
            elif len(shp) == 2:
                arr = data[0]
                # Heuristic: scores are in [0,1], classes typically > 1
                if scores is None and np.max(arr) <= 1.0:
                    scores = arr
                elif classes is None and np.max(arr) > 1.0:
                    classes = arr
            elif len(shp) == 1 and shp[0] == 1 and num is None:
                num = int(np.squeeze(data).astype(np.int32))

    # Final fallbacks
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores is None:
        scores = np.zeros((len(boxes),), dtype=np.float32)
    if classes is None:
        classes = np.zeros((len(boxes),), dtype=np.float32)
    if num is None:
        num = min(len(scores), len(boxes), len(classes))

    # Truncate arrays to num detections
    boxes = boxes[:num]
    scores = scores[:num]
    classes = classes[:num]
    return boxes, classes, scores, num

def main():
    # 1. Setup: load labels and initialize TFLite interpreter
    labels = load_labels(LABEL_PATH)
    label_offset = get_label_offset(labels)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assume single input tensor
    in_height, in_width = input_details[0]['shape'][1], input_details[0]['shape'][2]
    # Some models use NHWC [1, h, w, c]; if not, adjust accordingly
    if len(input_details[0]['shape']) == 4:
        _, in_height, in_width, _ = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_index = input_details[0]['index']

    # 2. Preprocessing: open input video and prepare output writer
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video at {INPUT_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = create_video_writer(OUTPUT_PATH, width, height, fps)

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Preprocessing: prepare input tensor
            input_tensor = preprocess_frame(frame, (in_height, in_width), input_dtype)

            # 3. Inference
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()

            # 4. Output handling: parse outputs and draw detections
            boxes, classes, scores, num = parse_outputs(interpreter, output_details)
            draw_detections(frame, boxes, classes, scores, labels, label_offset, CONFIDENCE_THRESHOLD)

            writer.write(frame)
            frame_count += 1

            # Optional: simple progress print every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                fps_proc = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Processed {frame_count} frames, approx {fps_proc:.2f} FPS")

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t_start
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    print(f"Done. Frames: {frame_count}, Time: {total_time:.2f}s, Avg FPS: {avg_fps:.2f}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()