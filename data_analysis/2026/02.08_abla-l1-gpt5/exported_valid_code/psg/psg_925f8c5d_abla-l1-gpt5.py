import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            lab = line.strip()
            if lab:
                labels.append(lab)
    return labels

def prepare_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def get_output_tensors(interpreter, output_details):
    # Map outputs using names when available; fallback to shapes
    boxes = classes = scores = num = None
    for od in output_details:
        idx = od['index']
        name = od.get('name', '')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='ignore')
        lname = str(name).lower()

        tensor = interpreter.get_tensor(idx)
        # Remove batch dimension if present
        if tensor.ndim > 1 and tensor.shape[0] == 1:
            tensor = np.squeeze(tensor, axis=0)

        if 'boxes' in lname:
            boxes = tensor
        elif 'scores' in lname:
            scores = tensor
        elif 'classes' in lname:
            classes = tensor
        elif 'num' in lname or 'count' in lname:
            num = int(tensor) if np.isscalar(tensor) else int(tensor.flatten()[0])

    # Fallback by shapes if names were not informative
    if boxes is None or scores is None or classes is None:
        for od in output_details:
            idx = od['index']
            tensor = interpreter.get_tensor(idx)
            if tensor.ndim > 1 and tensor.shape[0] == 1:
                tensor = np.squeeze(tensor, axis=0)
            shape = tensor.shape
            if boxes is None and len(shape) == 2 and shape[1] == 4:
                boxes = tensor
            elif scores is None and len(shape) == 1 and tensor.dtype == np.float32:
                scores = tensor
            elif classes is None and len(shape) == 1:
                classes = tensor
            elif num is None and (np.isscalar(tensor) or len(shape) == 0):
                num = int(tensor) if np.isscalar(tensor) else int(tensor.flatten()[0])

    # Final fallback for num
    if num is None:
        lengths = [len(x) for x in [boxes, scores, classes] if x is not None]
        num = min(lengths) if lengths else 0

    return boxes, classes, scores, num

def preprocess_frame(frame, input_shape, input_dtype):
    # input_shape expected: [1, height, width, 3]
    _, ih, iw, _ = input_shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    else:
        input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
    return input_data

def draw_detections(frame, boxes, classes, scores, num, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(int(num)):
        score = float(scores[i]) if scores is not None else 0.0
        if score < threshold:
            continue
        # boxes are in [ymin, xmin, ymax, xmax] normalized [0,1]
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, min(int(xmin * w), w - 1))
        y1 = max(0, min(int(ymin * h), h - 1))
        x2 = max(0, min(int(xmax * w), w - 1))
        y2 = max(0, min(int(ymax * h), h - 1))

        class_id = int(classes[i]) if classes is not None else -1
        label = str(class_id)
        if 0 <= class_id < len(labels):
            label = labels[class_id]

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        caption = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(th + 4, y1)
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + bl), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 1. Setup: Load labels and interpreter
    labels = load_labels(LABEL_PATH)
    interpreter, input_details, output_details = prepare_interpreter(MODEL_PATH)
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # 2. Input/Output handling setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video: {OUTPUT_PATH}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Preprocessing
            input_data = preprocess_frame(frame, input_shape, input_dtype)

            # 3. Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # 4. Output handling: parse tensors and draw on frame
            boxes, classes, scores, num = get_output_tensors(interpreter, output_details)
            draw_detections(frame, boxes, classes, scores, num, labels, CONFIDENCE_THRESHOLD)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

if __name__ == "__main__":
    main()