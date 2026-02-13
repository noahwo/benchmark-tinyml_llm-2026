import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def ensure_dir_for_file(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_input_size(interpreter):
    input_details = interpreter.get_input_details()[0]
    shape = input_details["shape"]
    # Expecting [1, height, width, channels]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected input tensor shape: {shape}")
    return int(shape[2]), int(shape[1])  # width, height (OpenCV uses width x height)

def preprocess_frame(frame_bgr, target_size, input_dtype):
    # Convert BGR (OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def classify_output_tensors(output_arrays):
    # output_arrays: list of np.ndarray squeezed
    boxes = None
    scores = None
    classes = None
    num = None
    one_dim_arrays = []
    for arr in output_arrays:
        if arr.ndim == 2 and arr.shape[1] == 4:
            boxes = arr
        elif (arr.ndim == 0) or (arr.ndim == 1 and arr.size == 1):
            num = float(arr.reshape(-1)[0])
        elif arr.ndim == 1:
            one_dim_arrays.append(arr)
    # Among 1D arrays, scores are in [0,1], classes are usually > 1
    for arr in one_dim_arrays:
        if arr.size == 0:
            continue
        if np.nanmax(arr) <= 1.0 + 1e-6:
            scores = arr
        else:
            classes = arr
    return boxes, classes, scores, num

def draw_detections(frame_bgr, detections, labels, offset):
    h, w = frame_bgr.shape[:2]
    palette = [
        (56, 56, 255), (151, 157, 255), (31, 112, 255),
        (29, 178, 255), (255, 219, 148), (255, 191, 0),
        (255, 255, 0), (0, 240, 240), (240, 0, 240), (240, 240, 0)
    ]
    boxes, classes, scores = detections
    num_dets = min(len(scores), len(classes), len(boxes))
    for i in range(num_dets):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        box = boxes[i]  # [ymin, xmin, ymax, xmax], normalized
        ymin = max(0, min(int(box[0] * h), h - 1))
        xmin = max(0, min(int(box[1] * w), w - 1))
        ymax = max(0, min(int(box[2] * h), h - 1))
        xmax = max(0, min(int(box[3] * w), w - 1))
        class_id_raw = int(classes[i])
        label_idx = class_id_raw + offset
        if 0 <= label_idx < len(labels):
            label_name = labels[label_idx]
        elif 0 <= class_id_raw < len(labels):
            label_name = labels[class_id_raw]
        else:
            label_name = f"id:{class_id_raw}"
        color = palette[class_id_raw % len(palette)] if len(palette) > 0 else (0, 255, 0)
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, 2)
        caption = f"{label_name} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (xmin, ymin - th - baseline), (xmin + tw, ymin), color, -1)
        cv2.putText(frame_bgr, caption, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    # 1. setup: interpreter initialization
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    labels = load_labels(label_path)
    # Determine if label file has a background/??? at index 0
    offset = 1 if (len(labels) > 0 and labels[0].strip().lower() in ("???", "background")) else 0

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]
    in_w, in_h = get_input_size(interpreter)  # width, height

    # 2. preprocessing: video IO initialization
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3 or np.isnan(fps):
        fps = 30.0
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read the first frame from the input video.")
    frame_h, frame_w = first_frame.shape[:2]
    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    frame_idx = 0

    start_time = time.time()
    # Process the first frame (already read) and then the rest
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 3. inference loop
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        input_data = preprocess_frame(frame_bgr, (in_w, in_h), input_dtype)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # 4. output handling: fetch outputs and visualize
        output_arrays = []
        for od in output_details:
            out = interpreter.get_tensor(od["index"])
            output_arrays.append(np.squeeze(out))
        boxes, classes, scores, num = classify_output_tensors(output_arrays)

        if boxes is None or classes is None or scores is None or num is None:
            # If outputs cannot be classified, skip drawing
            pass
        else:
            num_detections = int(num)
            boxes = boxes[:num_detections]
            classes = classes[:num_detections]
            scores = scores[:num_detections]
            draw_detections(frame_bgr, (boxes, classes, scores), labels, offset)

        writer.write(frame_bgr)
        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
            print(f"Processed {frame_idx} frames, avg FPS: {avg_fps:.2f}")

    cap.release()
    writer.release()
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(f"Done. Frames: {frame_idx}, Time: {total_time:.2f}s, Avg FPS: {avg_fps:.2f}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()