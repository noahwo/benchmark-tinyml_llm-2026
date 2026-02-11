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


def load_labels(label_path):
    # Supports plain label lists or "index label" format
    labels = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        idx = int(parts[0])
                        name = parts[1].strip()
                        while len(labels) <= idx:
                            labels.append("")
                        labels[idx] = name
                    else:
                        labels.append(line)
                else:
                    labels.append(line)
    except FileNotFoundError:
        labels = []
    return labels


def make_interpreter(model_path):
    # Try to initialize with num_threads (Raspberry Pi 4B has 4 cores).
    # Fallback to default if unsupported.
    try:
        num_threads = os.cpu_count() or 4
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except TypeError:
        return Interpreter(model_path=model_path)


def get_output_tensors(interpreter):
    # Map outputs to boxes, classes, scores, num_detections
    output_details = interpreter.get_output_details()

    # Attempt by name first (most reliable)
    name_map = {}
    for od in output_details:
        name = od.get("name", "").lower()
        if "box" in name:
            name_map["boxes"] = od["index"]
        elif "score" in name:
            name_map["scores"] = od["index"]
        elif "class" in name:
            name_map["classes"] = od["index"]
        elif "num" in name:
            name_map["num_detections"] = od["index"]

    # If names didn't resolve all, fall back to heuristics
    if len(name_map) < 4:
        for od in output_details:
            idx = od["index"]
            shape = od["shape"]
            dtype = od["dtype"]
            # Boxes: [1, N, 4]
            if tuple(shape[-2:]) == (None, None):
                pass  # dynamic shapes not expected here
            if "boxes" not in name_map and len(shape) == 3 and shape[-1] == 4:
                name_map["boxes"] = idx
            # Scores: [1, N]
            elif "scores" not in name_map and len(shape) == 2 and dtype == np.float32:
                name_map["scores"] = idx
            # Classes: [1, N]
            elif "classes" not in name_map and len(shape) == 2 and dtype in (np.float32, np.int64, np.int32):
                name_map["classes"] = idx
            # Num detections: [1]
            elif "num_detections" not in name_map and len(shape) == 1:
                name_map["num_detections"] = idx

    # Fetch tensors
    boxes = interpreter.get_tensor(name_map["boxes"])
    classes = interpreter.get_tensor(name_map["classes"])
    scores = interpreter.get_tensor(name_map["scores"])
    # Some models may omit num_detections; if so, infer from scores length
    if "num_detections" in name_map:
        num = interpreter.get_tensor(name_map["num_detections"])
        num = int(np.squeeze(num).astype(np.int32))
    else:
        num = scores.shape[1] if len(scores.shape) == 2 else scores.shape[0]

    # Squeeze batch dimension
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    return boxes, classes, scores, num


def draw_detections(frame, boxes, classes, scores, num, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue

        # TFLite SSD boxes are in normalized ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = max(0, int(xmin * w))
        y1 = max(0, int(ymin * h))
        x2 = min(w - 1, int(xmax * w))
        y2 = min(h - 1, int(ymax * h))

        class_id = int(classes[i])
        label_text = ""
        if labels and 0 <= class_id < len(labels) and labels[class_id]:
            label_text = labels[class_id]
        else:
            label_text = f"id:{class_id}"
        caption = f"{label_text} {score:.2f}"

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - baseline - 4), (x1 + text_w + 4, y1), color, thickness=-1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    # 1) SETUP
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    labels = load_labels(LABEL_PATH)

    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    input_shape = input_details["shape"]
    # Expect shape [1, height, width, 3]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    input_dtype = input_details["dtype"]

    # Video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or np.isnan(in_fps) or in_fps <= 1.0:
        in_fps = 30.0  # Fallback if FPS is unavailable
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # 2) PREPROCESSING + 3) INFERENCE + 4) OUTPUT HANDLING
    frame_count = 0
    total_inference_time = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocessing: BGR -> RGB, resize to model input
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

            if input_dtype == np.uint8:
                input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
            else:
                # Assume float model expects [0,1]
                input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            interpreter.set_tensor(input_index, input_tensor)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            inference_time = (time.time() - t0) * 1000.0  # ms
            total_inference_time += inference_time

            boxes, classes, scores, num = get_output_tensors(interpreter)

            # Draw detections on original BGR frame
            draw_detections(frame_bgr, boxes, classes, scores, num, labels, CONFIDENCE_THRESHOLD)

            # Optional: overlay performance
            cv2.putText(
                frame_bgr,
                f"Inference: {inference_time:.1f} ms",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame_bgr)

    finally:
        cap.release()
        writer.release()

    if frame_count > 0:
        avg_ms = total_inference_time / frame_count
        print(f"Processed {frame_count} frames. Average inference time: {avg_ms:.2f} ms/frame")
        print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()