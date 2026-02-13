import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    return labels

def select_label(labels, class_id):
    # class_id may be float; convert to int
    cid = int(class_id)
    if not labels:
        return f"id:{cid}"
    # If label file starts with background marker, assume 1-based class indices
    first = labels[0].strip().lower()
    has_background = first in ("???", "background", "bg")
    idx = cid if has_background else cid - 1
    if 0 <= idx < len(labels):
        return labels[idx]
    # Fallback if out of bounds
    return f"id:{cid}"

def preprocess(frame, input_size, input_dtype):
    ih, iw = input_size
    # Resize and convert BGR (OpenCV) to RGB (model expectation)
    resized = cv2.resize(frame, (iw, ih))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.uint8:
        tensor = rgb.astype(np.uint8)
    else:
        # Float model: normalize to [0,1]
        tensor = (rgb.astype(np.float32) / 255.0).astype(np.float32)
    # Add batch dimension
    return np.expand_dims(tensor, axis=0)

def get_output_tensors(interpreter):
    output_details = interpreter.get_output_details()
    boxes = classes = scores = num = None

    # Try to use tensor names if provided
    for od in output_details:
        name = od.get('name', '')
        data = interpreter.get_tensor(od['index'])
        if 'detection_boxes' in name:
            boxes = data[0] if data.ndim == 3 else data.reshape(-1, 4)
        elif 'detection_classes' in name:
            classes = data[0] if data.ndim > 1 else data
        elif 'detection_scores' in name:
            scores = data[0] if data.ndim > 1 else data
        elif 'num_detections' in name:
            num = int(data.flatten()[0])

    # Fallback heuristics if names were not informative
    if boxes is None or classes is None or scores is None or num is None:
        tensors = [interpreter.get_tensor(od['index']) for od in output_details]
        # Identify boxes by last dim 4
        for t in tensors:
            if t.ndim >= 2 and t.shape[-1] == 4:
                boxes = t[0] if t.ndim == 3 else t.reshape(-1, 4)
        # Remaining arrays likely classes, scores, num
        candidates = [t for t in tensors if not (t.ndim >= 2 and t.shape[-1] == 4)]
        # num: scalar or shape [1]
        for t in candidates:
            if t.size == 1:
                num = int(t.flatten()[0])
        # classes and scores are 1 x N or (N,)
        v = [t[0] if t.ndim == 2 and t.shape[0] == 1 else (t if t.ndim == 1 else None) for t in candidates if t.size > 1]
        v = [x for x in v if x is not None]
        # Heuristic: values in [0,1] -> scores
        for arr in v:
            if np.issubdtype(arr.dtype, np.floating) and np.all((arr >= 0) & (arr <= 1)):
                scores = arr
        # The other one -> classes
        for arr in v:
            if not np.shares_memory(arr, scores):
                classes = arr

    # Ensure num is valid
    if num is None:
        num = len(scores) if scores is not None else (len(classes) if classes is not None else 0)

    # Slice to num detections
    if boxes is not None:
        boxes = boxes[:num]
    if classes is not None:
        classes = classes[:num]
    if scores is not None:
        scores = scores[:num]

    return boxes, classes, scores, num

def draw_detections(frame, boxes, classes, scores, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        # Boxes in normalized ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))
        color = (0, 200, 0)  # Green
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label_text = select_label(labels, classes[i])
        caption = f"{label_text}: {score:.2f}"
        # Background for text
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline)
        cv2.rectangle(frame, (x1, y_text), (x1 + tw, y_text + th + baseline), color, -1)
        cv2.putText(frame, caption, (x1, y_text + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def main():
    # 1. Setup: model/load interpreter
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

    labels = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details:
        raise RuntimeError("Interpreter has no input details.")
    input_index = input_details[0]['index']
    input_dtype = input_details[0]['dtype']
    # Expect input shape: [1, height, width, 3]
    input_shape = input_details[0]['shape']
    if len(input_shape) != 4 or input_shape[-1] != 3:
        raise RuntimeError(f"Unexpected input tensor shape: {input_shape}")
    input_height, input_width = int(input_shape[1]), int(input_shape[2])

    # 2. Preprocessing: open video and prepare writer
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 1e-2:
        fps = 30.0  # fallback

    ensure_dir(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 2. Preprocessing: prepare input tensor
            input_tensor = preprocess(frame, (input_height, input_width), input_dtype)

            # 3. Inference
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()

            # 4. Output handling: parse outputs and draw on frame
            boxes, classes, scores, num = get_output_tensors(interpreter)
            if boxes is None or classes is None or scores is None:
                raise RuntimeError("Failed to retrieve detection outputs from the model.")
            annotated = draw_detections(frame, boxes, classes, scores, labels, CONFIDENCE_THRESHOLD)

            writer.write(annotated)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s (avg {avg_fps:.2f} FPS)")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()