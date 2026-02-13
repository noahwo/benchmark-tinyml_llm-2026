import os
import time
import numpy as np
import cv2

# Per instruction: use ai_edge_litert Interpreter
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5


def load_labels(path):
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support formats like "0 person", "0: person", or just "person"
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[1].strip()
                labels.append(name)
            elif line[:1].isdigit():
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1].strip())
                else:
                    labels.append(line.strip())
            else:
                labels.append(line.strip())
    return labels


def get_label_name(class_id, labels):
    if not labels:
        return str(class_id)
    # Handle off-by-one depending on label map (some start with '???')
    if 0 <= class_id < len(labels):
        name = labels[class_id]
        if name == "???":
            # Try class_id+1
            if 0 <= class_id + 1 < len(labels):
                return labels[class_id + 1]
        return name
    elif 0 <= class_id - 1 < len(labels):
        return labels[class_id - 1]
    return str(class_id)


def prepare_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Assume single input
    in_det = input_details[0]
    in_index = in_det["index"]
    in_dtype = in_det["dtype"]
    in_shape = in_det["shape"]  # [1, height, width, 3]
    return in_index, in_dtype, in_shape, output_details


def preprocess_frame(frame, input_shape, input_dtype):
    # input_shape: [1, height, width, 3]
    h, w = int(input_shape[1]), int(input_shape[2])
    # Convert BGR->RGB and resize
    resized = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)
    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    elif input_dtype == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        # Fallback: cast to required dtype without normalization
        input_data = input_data.astype(input_dtype)
    return input_data


def parse_outputs(interpreter, output_details):
    # Attempt to identify boxes, classes, scores, num_detections from output_details
    boxes_idx = None
    classes_idx = None
    scores_idx = None
    num_idx = None

    for i, det in enumerate(output_details):
        shape = det.get("shape", [])
        dtype = det.get("dtype", np.float32)
        size = int(np.prod(shape)) if len(shape) > 0 else 1

        if len(shape) >= 2 and shape[-1] == 4:
            boxes_idx = i
        elif size == 1:
            num_idx = i
        else:
            # Heuristic: scores are float, classes often float/int but we will assign after scores
            if dtype == np.float32:
                # prefer scores as float
                if scores_idx is None:
                    scores_idx = i
            else:
                if classes_idx is None:
                    classes_idx = i

    # If classes_idx not found but we have two float outputs, pick the non-box, non-num one left
    if classes_idx is None:
        for i, det in enumerate(output_details):
            if i in (boxes_idx, scores_idx, num_idx):
                continue
            classes_idx = i
            break

    def get_tensor_by_idx(idx):
        if idx is None:
            return None
        return interpreter.get_tensor(output_details[idx]["index"])

    boxes = get_tensor_by_idx(boxes_idx)
    classes = get_tensor_by_idx(classes_idx)
    scores = get_tensor_by_idx(scores_idx)
    num = get_tensor_by_idx(num_idx)

    # Normalize shapes to common format
    # Expected shapes: boxes [1, N, 4]; classes [1, N]; scores [1, N]; num [1]
    if boxes is not None and boxes.ndim == 2 and boxes.shape[-1] == 4:
        boxes = np.expand_dims(boxes, 0)
    if classes is not None and classes.ndim == 1:
        classes = np.expand_dims(classes, 0)
    if scores is not None and scores.ndim == 1:
        scores = np.expand_dims(scores, 0)
    if num is not None and num.size == 1:
        num = float(num.flatten()[0])
    elif scores is not None:
        num = scores.shape[1]
    else:
        num = 0

    # Convert to usable numpy arrays
    boxes = boxes if boxes is not None else np.zeros((1, 0, 4), dtype=np.float32)
    classes = classes if classes is not None else np.zeros((1, 0), dtype=np.float32)
    scores = scores if scores is not None else np.zeros((1, 0), dtype=np.float32)

    return boxes, classes, scores, int(num)


def draw_detections(frame, boxes, classes, scores, labels, threshold, running_map=None):
    h, w = frame.shape[:2]
    for i in range(boxes.shape[1]):
        score = float(scores[0, i])
        if score < threshold:
            continue
        cls_id = int(classes[0, i])
        label = get_label_name(cls_id, labels)
        ymin, xmin, ymax, xmax = boxes[0, i]
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}: {score*100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay mAP (proxy) on frame if provided
    if running_map is not None:
        text = f"mAP: {running_map:.3f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        pad = 6
        cv2.rectangle(frame, (10, 10), (10 + tw + 2 * pad, 10 + th + 2 * pad), (0, 0, 0), -1)
        cv2.putText(frame, text, (10 + pad, 10 + th + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Setup TFLite interpreter
    interpreter = prepare_interpreter(model_path)
    in_index, in_dtype, in_shape, out_details = get_io_details(interpreter)

    # Video IO
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open input video: {input_path}")
        return

    # Read first frame to setup writer
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: failed to read the first frame from input video.")
        cap.release()
        return

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback if FPS not available
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: cannot open output video for writing: {output_path}")
        cap.release()
        return

    # Processing loop
    total_det_conf = 0.0  # For proxy mAP: sum of confidences of detections above threshold
    total_det_count = 0   # For proxy mAP: count of detections above threshold

    # Process the already-read first frame then continue
    frames_processed = 0
    start_time = time.time()

    def process_and_write(curr_frame):
        nonlocal total_det_conf, total_det_count
        input_data = preprocess_frame(curr_frame, in_shape, in_dtype)
        interpreter.set_tensor(in_index, input_data)
        interpreter.invoke()
        boxes, classes, scores, num = parse_outputs(interpreter, out_details)

        # Update proxy mAP accumulators
        if scores.size > 0:
            valid = scores[0] >= confidence_threshold
            if np.any(valid):
                total_det_conf += float(np.sum(scores[0][valid]))
                total_det_count += int(np.sum(valid))

        # Compute running proxy mAP
        map_proxy = (total_det_conf / total_det_count) if total_det_count > 0 else 0.0

        # Draw and write
        draw_detections(curr_frame, boxes, classes, scores, labels, confidence_threshold, running_map=map_proxy)
        writer.write(curr_frame)

    process_and_write(frame)
    frames_processed += 1

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        process_and_write(frame)
        frames_processed += 1

    elapsed = time.time() - start_time
    cap.release()
    writer.release()

    # Final summary
    fps_proc = frames_processed / elapsed if elapsed > 0 else 0.0
    map_proxy_final = (total_det_conf / total_det_count) if total_det_count > 0 else 0.0
    print(f"Processed {frames_processed} frames in {elapsed:.2f}s ({fps_proc:.2f} FPS).")
    print(f"Saved annotated video to: {output_path}")
    print(f"mAP (proxy without ground truth): {map_proxy_final:.4f}")


if __name__ == "__main__":
    main()