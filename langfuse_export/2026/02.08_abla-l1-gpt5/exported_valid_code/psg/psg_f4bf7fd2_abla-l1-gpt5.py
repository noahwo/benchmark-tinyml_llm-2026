import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =============================================================================
# Configuration Parameters
# =============================================================================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# =============================================================================
# Utilities
# =============================================================================
def load_labels(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                labels.append(name)
    return labels

def ensure_dir_for_file(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_label_name(labels, class_id):
    # Handle possible label offset differences ('???' placeholder at index 0)
    if not labels:
        return f"id:{int(class_id)}"
    if labels[0] == "???":
        idx = int(class_id)
    else:
        idx = int(class_id) - 1
    if 0 <= idx < len(labels):
        return labels[idx]
    return f"id:{int(class_id)}"

# =============================================================================
# 1. Setup: Interpreter initialization
# =============================================================================
def setup_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# =============================================================================
# 2. Preprocessing
# =============================================================================
def preprocess_frame(frame, input_details):
    # Assumes single input tensor
    in_info = input_details[0]
    _, in_h, in_w, in_c = in_info["shape"]
    dtype = in_info["dtype"]

    # Convert BGR (OpenCV) to RGB, resize to model input size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    if dtype == np.uint8:
        input_tensor = resized.astype(np.uint8)
    else:
        # Generic normalization for float32 models
        input_tensor = resized.astype(np.float32) / 255.0

    # Expand to NHWC batch
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

# =============================================================================
# 3. Inference
# =============================================================================
def run_inference(interpreter, input_details, output_details, input_tensor):
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    # Standard TFLite SSD output order:
    # 0: detection_boxes [1, num, 4] (ymin, xmin, ymax, xmax) normalized
    # 1: detection_classes [1, num]
    # 2: detection_scores [1, num]
    # 3: num_detections [1]
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    num = int(interpreter.get_tensor(output_details[3]["index"])[0])
    return boxes, classes, scores, num

# =============================================================================
# 4. Output Handling: Visualization and video writing
# =============================================================================
def draw_detections(frame, boxes, classes, scores, num, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        left = int(max(0, x_min * w))
        top = int(max(0, y_min * h))
        right = int(min(w - 1, x_max * w))
        bottom = int(min(h - 1, y_max * h))

        label_name = get_label_name(labels, classes[i])
        caption = f"{label_name}: {score:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (left, max(0, top - th - baseline - 3)),
                      (left + tw + 4, top), (0, 255, 0), thickness=-1)
        # Put label text
        cv2.putText(frame, caption, (left + 2, top - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def main():
    ensure_dir_for_file(OUTPUT_PATH)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Setup interpreter
    interpreter, input_details, output_details = setup_interpreter(MODEL_PATH)
    in_h = input_details[0]["shape"][1]
    in_w = input_details[0]["shape"][2]

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {INPUT_PATH}")
        return

    # Prepare output video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    print("Starting inference...")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH}")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()

            # Preprocess
            input_tensor = preprocess_frame(frame, input_details)

            # Inference
            boxes, classes, scores, num = run_inference(
                interpreter, input_details, output_details, input_tensor
            )

            # Draw detections and write frame
            annotated = draw_detections(
                frame, boxes, classes, scores, num, labels, CONFIDENCE_THRESHOLD
            )
            writer.write(annotated)

            frame_count += 1
            t_end = time.time()
            if frame_count % 30 == 0:
                elapsed = t_end - t0
                fps_runtime = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Processed {frame_count} frames, approx FPS: {fps_runtime:.2f}")
    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    overall_fps = frame_count / total_time if total_time > 0 else 0.0
    print(f"Done. Total frames: {frame_count}, Total time: {total_time:.2f}s, Avg FPS: {overall_fps:.2f}")

if __name__ == "__main__":
    main()