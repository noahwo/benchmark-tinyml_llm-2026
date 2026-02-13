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

def load_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some label maps may be "id label". Keep only the label text
            parts = line.split(maxsplit=1)
            label = parts[-1]
            labels.append(label)
    return labels

def get_label_name(labels, class_id):
    # Handle both 0-based and 1-based label files gracefully
    idx = int(class_id)
    if 0 <= idx < len(labels) and labels[idx] not in ("???", ""):
        return labels[idx]
    idx_minus = idx - 1
    if 0 <= idx_minus < len(labels) and labels[idx_minus] not in ("???", ""):
        return labels[idx_minus]
    return f"id:{int(class_id)}"

def prepare_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    # Determine input shape and dtype
    _, input_height, input_width, _ = input_details["shape"]
    input_dtype = input_details["dtype"]
    quant_params = input_details.get("quantization", (0.0, 0))
    return interpreter, input_height, input_width, input_dtype, quant_params, output_details

def preprocess_frame(frame_bgr, input_width, input_height, input_dtype):
    # Convert BGR to RGB as most TFLite models expect RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height))
    if input_dtype == np.float32:
        input_tensor = resized.astype(np.float32) / 255.0
    else:
        input_tensor = resized.astype(np.uint8)
    return np.expand_dims(input_tensor, axis=0)

def parse_detections(interpreter, output_details):
    # Retrieve outputs and identify tensors
    boxes = None
    classes = None
    scores = None
    num_detections = None

    for det in output_details:
        out = interpreter.get_tensor(det["index"])
        name = det.get("name", "").lower()
        shape = out.shape

        # Try name-based identification first
        if "box" in name:
            boxes = out[0] if out.ndim == 3 else out
            continue
        if "score" in name:
            scores = out[0] if out.ndim == 2 else out
            continue
        if "class" in name:
            classes = out[0].astype(np.int32) if out.ndim == 2 else out.astype(np.int32)
            continue
        if "num" in name:
            num_detections = int(out.flatten()[0])
            continue

        # Fallback to shape-based identification
        if boxes is None and out.ndim == 3 and out.shape[-1] == 4:
            boxes = out[0]
        elif out.ndim == 2 and out.shape[0] == 1:
            if scores is None and np.max(out) <= 1.0 + 1e-6:
                scores = out[0]
            elif classes is None:
                classes = out[0].astype(np.int32)
        elif out.size == 1 and num_detections is None:
            num_detections = int(out.flatten()[0])

    # Reasonable defaults if num_detections not explicitly provided
    if boxes is not None and num_detections is None:
        num_detections = boxes.shape[0]
    if scores is not None and num_detections is None:
        num_detections = scores.shape[0]
    if classes is not None and num_detections is None:
        num_detections = classes.shape[0]

    return boxes, classes, scores, num_detections

def draw_detections(frame_bgr, boxes, classes, scores, num_detections, labels, threshold):
    h, w = frame_bgr.shape[:2]
    for i in range(int(num_detections)):
        score = float(scores[i]) if scores is not None else 0.0
        if score < threshold:
            continue

        # Boxes are in [ymin, xmin, ymax, xmax] normalized coordinates
        if boxes is None:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, int(xmin * w))
        y1 = max(0, int(ymin * h))
        x2 = min(w - 1, int(xmax * w))
        y2 = min(h - 1, int(ymax * h))

        cls_id = int(classes[i]) if classes is not None else -1
        label_text = get_label_name(labels, cls_id)
        caption = f"{label_text}: {score:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 200, 0), cv2.FILLED)
        cv2.putText(frame_bgr, caption, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Step 1: Setup - load labels and initialize interpreter
    labels = load_labels(LABEL_PATH)
    interpreter, in_h, in_w, in_dtype, _, output_details = prepare_interpreter(MODEL_PATH)

    # Initialize video I/O
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0:
        src_fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, src_fps, (frame_width, frame_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    input_index = interpreter.get_input_details()[0]["index"]

    # Processing loop
    frame_count = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 2: Preprocessing
        input_tensor = preprocess_frame(frame, in_w, in_h, in_dtype)

        # Step 3: Inference
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

        # Step 4: Output handling
        boxes, classes, scores, num_detections = parse_detections(interpreter, output_details)
        if boxes is not None and scores is not None and classes is not None and num_detections is not None:
            draw_detections(frame, boxes, classes, scores, num_detections, labels, CONFIDENCE_THRESHOLD)

        writer.write(frame)
        frame_count += 1

    t1 = time.time()
    elapsed = max(1e-9, t1 - t0)
    fps = frame_count / elapsed

    cap.release()
    writer.release()

    print(f"Processing completed.")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {fps:.2f}")

if __name__ == "__main__":
    main()