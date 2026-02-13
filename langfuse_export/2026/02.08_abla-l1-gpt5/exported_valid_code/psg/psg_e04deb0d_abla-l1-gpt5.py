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

def load_labels(path):
    labels = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"Warning: Failed to load labels from {path}: {e}")
    return labels

def get_label_name(labels, class_id):
    # Many TFLite label files have "???" as first label; classes often start at 1
    if not labels:
        return str(class_id)
    if 0 <= class_id < len(labels):
        return labels[class_id]
    if 0 <= (class_id - 1) < len(labels):
        return labels[class_id - 1]
    return str(class_id)

def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def find_output_indices(output_details):
    # Attempt to identify outputs by name; fallback to shape heuristics
    indices = {"boxes": None, "classes": None, "scores": None, "count": None}
    # First try by name
    for i, od in enumerate(output_details):
        name = od.get("name", "").lower()
        shape = od.get("shape", [])
        if "box" in name or "boxes" in name:
            indices["boxes"] = i
        elif "class" in name or "classes" in name:
            indices["classes"] = i
        elif "score" in name or "scores" in name:
            indices["scores"] = i
        elif "num" in name or "count" in name or (isinstance(shape, (list, np.ndarray)) and np.prod(shape) == 1):
            indices["count"] = i

    # Fallback by shape if needed
    if indices["boxes"] is None or indices["scores"] is None or indices["classes"] is None:
        for i, od in enumerate(output_details):
            shape = od.get("shape", [])
            if len(shape) == 3 and shape[-1] == 4:
                indices["boxes"] = i
        for i, od in enumerate(output_details):
            shape = od.get("shape", [])
            if len(shape) >= 2 and shape[-1] != 4:
                # Could be classes or scores; distinguish by dtype (classes often float32 but will be cast to int)
                # We'll select both by elimination next
                pass
        # If still missing, try to assign remaining by shapes (1,N) assuming boxes already found
        remaining = [i for i in range(len(output_details)) if i not in (indices["boxes"], indices["count"])]
        # Heuristics: of remaining, both classes and scores typically have same shape; assign arbitrarily then swap by dtype ranges later
        if indices["scores"] is None and remaining:
            indices["scores"] = remaining[0]
        if indices["classes"] is None and len(remaining) > 1:
            indices["classes"] = remaining[1]

    return indices

def preprocess(frame_bgr, input_size, input_dtype, quant_params):
    ih, iw = input_size
    # Convert BGR to RGB and resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        scale, zero_point = (0.0, 0)
        if isinstance(quant_params, (list, tuple)) and len(quant_params) == 2:
            scale, zero_point = quant_params
        # Map to "real-world" [0,1] then quantize using scale/zero_point when available
        if scale and scale > 0.0:
            real = resized.astype(np.float32) / 255.0
            q = real / scale + zero_point
            q = np.clip(np.round(q), 0, 255).astype(np.uint8)
            input_data = q
        else:
            input_data = resized.astype(np.uint8)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def draw_detections(frame_bgr, detections, labels, threshold):
    h, w = frame_bgr.shape[:2]
    boxes, classes, scores, count = detections
    if boxes is None or scores is None or classes is None:
        return frame_bgr

    # Flatten batch dimension if present
    if len(boxes.shape) == 3:
        boxes = boxes[0]
    if len(scores.shape) == 2:
        scores = scores[0]
    if len(classes.shape) == 2:
        classes = classes[0]
    if count is not None:
        if hasattr(count, "__len__"):
            count = int(np.squeeze(count).astype(np.int32))
        else:
            count = int(count)
    else:
        count = len(scores)

    for i in range(count):
        score = float(scores[i])
        if score < threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        # Scale to absolute pixel coordinates
        left = int(max(0, min(1, x_min)) * w)
        top = int(max(0, min(1, y_min)) * h)
        right = int(max(0, min(1, x_max)) * w)
        bottom = int(max(0, min(1, y_max)) * h)

        class_id = int(classes[i])
        label = get_label_name(labels, class_id)
        caption = f"{label}: {score*100:.1f}%"

        # Draw rectangle
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

        # Draw label background
        text_scale = 0.5
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        text_x = max(0, left)
        text_y = max(th + baseline + 2, top)
        cv2.rectangle(frame_bgr, (text_x, text_y - th - baseline - 2), (text_x + tw + 2, text_y + 2), color, -1)
        cv2.putText(frame_bgr, caption, (text_x + 1, text_y - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

    return frame_bgr

def main():
    # Setup: load labels and initialize interpreter
    labels = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        raise RuntimeError("Interpreter has no input details. Check the model.")
    input_info = input_details[0]
    input_shape = input_info["shape"]
    # Expect shape: [1, height, width, channels]
    if len(input_shape) != 4:
        raise RuntimeError(f"Unexpected input tensor shape: {input_shape}")
    input_height, input_width = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_info["dtype"]
    quant_params = input_info.get("quantization", (0.0, 0))

    out_idx_map = find_output_indices(output_details)
    boxes_idx = output_details[out_idx_map["boxes"]]["index"] if out_idx_map["boxes"] is not None else None
    classes_idx = output_details[out_idx_map["classes"]]["index"] if out_idx_map["classes"] is not None else None
    scores_idx = output_details[out_idx_map["scores"]]["index"] if out_idx_map["scores"] is not None else None
    count_idx = output_details[out_idx_map["count"]]["index"] if out_idx_map["count"] is not None else None

    # Video I/O setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0  # Fallback FPS
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w == 0 or frame_h == 0:
        # Fallback: try to read one frame to infer size
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Unable to read frames to determine video size.")
        frame_h, frame_w = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ensure_dir(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    print("Starting inference...")
    print(f"Model: {MODEL_PATH}")
    print(f"Input size (model): {input_width}x{input_height}, dtype: {input_dtype}")
    print(f"Input video: {INPUT_PATH} ({frame_w}x{frame_h} @ {fps:.2f} FPS)")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    frame_index = 0
    t_last = time.time()
    avg_fps = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocessing
            input_data = preprocess(frame, (input_height, input_width), input_dtype, quant_params)
            interpreter.set_tensor(input_details[0]["index"], input_data)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            infer_ms = (time.time() - t0) * 1000.0

            # Output handling
            boxes = interpreter.get_tensor(boxes_idx) if boxes_idx is not None else None
            classes = interpreter.get_tensor(classes_idx) if classes_idx is not None else None
            scores = interpreter.get_tensor(scores_idx) if scores_idx is not None else None
            count = interpreter.get_tensor(count_idx) if count_idx is not None else None

            frame = draw_detections(frame, (boxes, classes, scores, count), labels, CONFIDENCE_THRESHOLD)

            # FPS estimation (rolling)
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            if avg_fps is None:
                avg_fps = inst_fps
            else:
                avg_fps = 0.9 * avg_fps + 0.1 * inst_fps

            # Overlay performance info
            perf_text = f"FPS: {avg_fps:.1f} | Inference: {infer_ms:.1f} ms"
            cv2.putText(frame, perf_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 255), 1, cv2.LINE_AA)

            writer.write(frame)
            frame_index += 1

            # Optional: print periodic progress
            if frame_index % 50 == 0:
                print(f"Processed {frame_index} frames. {perf_text}")

    finally:
        cap.release()
        writer.release()

    print(f"Done. Processed {frame_index} frames. Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()