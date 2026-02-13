import os
import time
import numpy as np
import cv2

# 1) setup: Import and initialize the TFLite interpreter
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5


def load_labels(path):
    labels = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Handle either "id label" or "label" formats
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1].strip()
                    while len(labels) <= idx:
                        labels.append("")
                    labels[idx] = name
                else:
                    labels.append(line)
    except FileNotFoundError:
        pass
    return labels


def prepare_input(frame_bgr, input_details):
    # 2) preprocessing: convert to model's expected input format
    h, w = input_details[0]["shape"][1], input_details[0]["shape"][2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (w, h))
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        input_tensor = resized.astype(np.uint8)
    else:
        # Default to float32 normalized [0,1]
        input_tensor = resized.astype(np.float32) / 255.0

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def parse_detections(interpreter, output_details):
    # 3) inference output parsing: obtain boxes, classes, scores, and count
    arrays = []
    for d in output_details:
        arr = interpreter.get_tensor(d["index"])
        arrays.append(np.squeeze(arr))

    boxes = None
    classes = None
    scores = None
    num = None

    for arr in arrays:
        if arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = arr  # [N,4] y_min, x_min, y_max, x_max (normalized)
        elif arr.ndim == 1:
            if arr.size == 1:
                num = int(round(float(arr.reshape(-1)[0])))
            else:
                # Heuristic to differentiate scores vs classes
                if arr.dtype == np.float32 and np.nanmax(arr) <= 1.0001:
                    scores = arr.astype(np.float32)
                else:
                    classes = arr.astype(np.int32)

    # Fallback if needed (some models might output different order/dtypes)
    if classes is None or scores is None or boxes is None:
        # Attempt to infer by sorting by shapes/types
        for arr in arrays:
            if boxes is None and arr.ndim == 2 and arr.shape[-1] == 4:
                boxes = arr
        one_d_arrays = [a for a in arrays if a.ndim == 1 and a.size > 1]
        if len(one_d_arrays) >= 2:
            # Choose scores as the array with max <= 1.0
            for a in one_d_arrays:
                if np.nanmax(a) <= 1.0001:
                    scores = a.astype(np.float32)
                else:
                    classes = a.astype(np.int32)
        if num is None:
            # If num not provided, infer from scores length
            num = len(scores) if scores is not None else 0

    if classes is None or scores is None or boxes is None or num is None:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32), 0

    # Ensure consistent lengths
    n = min(num, len(scores), len(classes), len(boxes))
    return boxes[:n], classes[:n], scores[:n], n


def draw_detections(frame, boxes, classes, scores, labels, threshold=0.5):
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue

        y_min, x_min, y_max, x_max = boxes[i]
        x1 = max(0, int(x_min * w))
        y1 = max(0, int(y_min * h))
        x2 = min(w - 1, int(x_max * w))
        y2 = min(h - 1, int(y_max * h))

        class_id = int(classes[i]) if not np.isnan(classes[i]) else -1
        label_name = ""
        if 0 <= class_id < len(labels):
            label_name = labels[class_id]
        if not label_name or label_name == "???":
            label_name = f"id:{class_id}"

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    # Load labels
    labels = load_labels(label_path)

    # Setup interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]

    # Video I/O setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    frame_count = 0
    t0 = time.time()
    inf_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2) preprocessing
            input_tensor = prepare_input(frame, input_details)

            # 3) inference
            interpreter.set_tensor(input_index, input_tensor)
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer_end = time.time()
            inf_times.append(t_infer_end - t_infer_start)

            boxes, classes, scores, num = parse_detections(interpreter, output_details)

            # 4) output handling (draw and write)
            draw_detections(frame, boxes, classes, scores, labels, threshold=confidence_threshold)
            writer.write(frame)

            frame_count += 1
            if frame_count % 30 == 0:
                avg_inf = (sum(inf_times[-30:]) / min(30, len(inf_times))) if inf_times else 0.0
                print(f"Processed {frame_count} frames | Avg inference {avg_inf*1000:.1f} ms")

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    avg_inf_ms = (sum(inf_times) / len(inf_times) * 1000.0) if inf_times else 0.0
    print(f"Done. Frames: {frame_count}, Total time: {total_time:.2f}s, Avg inference: {avg_inf_ms:.1f} ms/frame")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()