# Application: TFLite object detection
# Target device: Raspberry Pi 4B

import os
import time
import numpy as np
import cv2

# 1. setup: Import and initialize the TFLite interpreter
try:
    from ai_edge_litert.interpreter import Interpreter
except Exception as e:
    raise RuntimeError("Failed to import ai_edge_litert.Interpreter. Ensure 'ai-edge-litert' is installed.") from e

# Configuration parameters (provided)
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                # Handle potential "id name" format or plain name per line
                parts = name.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1])
                else:
                    labels.append(name)
    return labels

def ensure_output_dir(path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

def setup_interpreter(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details:
        raise RuntimeError("Interpreter has no input details.")
    return interpreter, input_details[0], output_details

def get_output_tensors(interpreter, output_details):
    # Map standard detection outputs by name if possible
    boxes, classes, scores, count = None, None, None, None
    # Try to find by common name patterns first
    for det in output_details:
        name = det.get("name", "").lower()
        idx = det["index"]
        tensor = interpreter.get_tensor(idx)
        if "detection_boxes" in name or ":0" in name and tensor.ndim == 3 and tensor.shape[-1] == 4:
            boxes = tensor
        elif "detection_classes" in name:
            classes = tensor
        elif "detection_scores" in name:
            scores = tensor
        elif "num_detections" in name:
            count = tensor

    # Fallback by heuristics (SSD models typically output 4 tensors)
    if boxes is None or classes is None or scores is None:
        # Sort by shape characteristics if names aren't informative
        for det in output_details:
            idx = det["index"]
            tensor = interpreter.get_tensor(idx)
            shp = tensor.shape
            if tensor.ndim >= 2 and shp[-1] == 4 and boxes is None:
                boxes = tensor
            elif tensor.ndim >= 2 and classes is None and np.issubdtype(tensor.dtype, np.integer):
                classes = tensor
            elif tensor.ndim >= 2 and scores is None and np.issubdtype(tensor.dtype, np.floating):
                scores = tensor
            elif tensor.size == 1 and count is None:
                count = tensor

    # Squeeze batch dimension if present
    if boxes is not None and boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if classes is not None and classes.ndim == 2 and classes.shape[0] == 1:
        classes = classes[0]
    if scores is not None and scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if count is not None and count.size == 1:
        count = int(np.squeeze(count))
    else:
        count = len(scores) if scores is not None else 0

    return boxes, classes, scores, count

def preprocess_frame(frame, input_detail):
    # 2. preprocessing
    # Convert BGR (OpenCV) to RGB as most TFLite models expect RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Input tensor expected shape: [1, height, width, channels]
    _, in_h, in_w, _ = input_detail["shape"]
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    dtype = input_detail["dtype"]
    # Determine quantization parameters if available
    scale, zero_point = 0.0, 0
    if "quantization" in input_detail and isinstance(input_detail["quantization"], tuple):
        scale, zero_point = input_detail["quantization"]

    if dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        # Quantized path (uint8/int8)
        if scale and scale > 0:
            # Normalize to [0,1], then quantize using (x/scale + zero_point)
            input_data = resized.astype(np.float32) / 255.0
            input_data = np.clip(np.round(input_data / scale + zero_point), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
        else:
            # No quantization parameters; pass raw bytes
            input_data = resized.astype(dtype)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def draw_detections(frame, boxes, classes, scores, count, labels, threshold):
    h, w = frame.shape[:2]
    # Dynamic styling
    thickness = max(2, (h + w) // 600)
    font_scale = max(0.5, min(1.0, (h * w) / (1280 * 720)))

    for i in range(count):
        score = float(scores[i]) if i < len(scores) else 0.0
        if score < threshold:
            continue

        cls_id = int(classes[i]) if i < len(classes) else -1

        # Label retrieval with robustness to 0/1-based indices
        label_name = "unknown"
        if 0 <= cls_id < len(labels):
            label_name = labels[cls_id]
        elif 0 <= (cls_id - 1) < len(labels):
            label_name = labels[cls_id - 1]

        # Boxes are in normalized ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        # Draw box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f"{label_name}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def main():
    # Validate files
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")
    if not os.path.isfile(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Setup interpreter
    interpreter, input_detail, output_details = setup_interpreter(MODEL_PATH)
    input_index = input_detail["index"]

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ensure_output_dir(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer for: {OUTPUT_PATH}")

    # Processing loop
    frame_count = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 2. preprocessing
        input_data = preprocess_frame(frame, input_detail)

        # 3. inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # 4. output handling
        boxes, classes, scores, count = get_output_tensors(interpreter, output_details)
        if boxes is None or classes is None or scores is None:
            # If outputs are not as expected, write original frame and continue
            writer.write(frame)
            continue

        draw_detections(frame, boxes, classes, scores, count, labels, CONFIDENCE_THRESHOLD)

        # Optional overlay: FPS
        elapsed = time.time() - t0
        fps_text = f"FPS: {frame_count / elapsed:.2f}" if elapsed > 0 else "FPS: --"
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

if __name__ == "__main__":
    main()