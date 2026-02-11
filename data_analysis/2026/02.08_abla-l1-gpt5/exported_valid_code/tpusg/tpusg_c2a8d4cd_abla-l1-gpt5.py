import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# ==============================
# Configuration parameters
# ==============================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# ==============================
# Utilities
# ==============================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print(f"Warning: label file not found at {path}. Using empty labels.")
        return labels
    with open(path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try formats: "id label", "id:label", or "label"
            if ":" in line:
                # id:label
                parts = line.split(":", 1)
                try:
                    key = int(parts[0].strip())
                    value = parts[1].strip()
                    labels[key] = value
                    continue
                except ValueError:
                    pass
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                # id label (label may include spaces)
                key = int(parts[0])
                value = " ".join(parts[1:])
                labels[key] = value
            else:
                # label only; assign incremental id starting at current idx
                labels[idx] = line
                idx += 1
    return labels

def preprocess_frame(frame_bgr, input_height, input_width, input_dtype):
    # Convert BGR (OpenCV) to RGB as most TFLite models expect RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.uint8:
        input_data = resized.astype(np.uint8)
    else:
        # Float32 path: scale to [0,1]
        input_data = resized.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(input_data, axis=0)

def get_output_tensors(interpreter):
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
    # Attempt to identify typical detection outputs by shapes
    boxes = None
    classes = None
    scores = None
    num = None
    for arr in outputs:
        arr_squeezed = np.squeeze(arr)
        if arr_squeezed.ndim == 2 and arr_squeezed.shape[-1] == 4:
            boxes = arr_squeezed  # [N,4] ymin, xmin, ymax, xmax (normalized)
        elif arr_squeezed.ndim == 1 and arr_squeezed.dtype in (np.float32, np.int32):
            # Could be classes or scores or num
            if arr_squeezed.size > 1:
                # Heuristic: scores typically float32 in [0,1]
                if arr_squeezed.dtype == np.float32 and np.all((arr_squeezed >= 0) & (arr_squeezed <= 1)):
                    scores = arr_squeezed
                else:
                    classes = arr_squeezed
            else:
                num = int(arr_squeezed[0]) if arr_squeezed.size == 1 else None
        elif arr_squeezed.ndim == 1 and arr_squeezed.size == 1:
            num = int(arr_squeezed[0])
    # Fallback if shapes aren't squeezed as expected (e.g., batch dimension present)
    if boxes is None or scores is None or classes is None:
        # Try with batch dimension assumptions
        for arr in outputs:
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr[0]
            elif arr.ndim == 2:
                # Could be scores or classes
                if arr.dtype == np.float32 and np.all((arr[0] >= 0) & (arr[0] <= 1)):
                    scores = arr[0]
                else:
                    classes = arr[0]
            elif arr.ndim == 1 and arr.size == 1:
                num = int(arr[0])
    return boxes, classes, scores, num

def draw_detections(frame, detections, labels, color_cache):
    h, w = frame.shape[:2]
    for det in detections:
        ymin, xmin, ymax, xmax, score, class_id = det
        # Clip to frame boundaries
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))
        cid = int(class_id)
        label = labels.get(cid, f"id:{cid}")
        color = color_cache.get(cid)
        if color is None:
            # Deterministic color by class id
            np.random.seed(cid)
            color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
            color_cache[cid] = color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, thickness=-1)
        cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def update_stats_and_compute_map(stats_by_class, detections):
    # stats_by_class: dict[class_id] -> list of confidences
    for det in detections:
        score = float(det[4])
        cid = int(det[5])
        stats_by_class.setdefault(cid, []).append(score)
    # Proxy mAP calculation: AP per class = mean(confidences for that class); mAP = mean over classes that have detections
    ap_values = [float(np.mean(v)) for v in stats_by_class.values() if len(v) > 0]
    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter with EdgeTPU delegate
    try:
        interpreter = Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load EdgeTPU delegate '{EDGETPU_SHARED_LIB}': {e}")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    _, input_height, input_width, _ = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    # Prepare output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # Stats for proxy mAP
    stats_by_class = {}
    color_cache = {}

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            input_data = preprocess_frame(frame_bgr, input_height, input_width, input_dtype)
            # Set input tensor
            interpreter.set_tensor(input_index, input_data)
            # Inference
            interpreter.invoke()
            # Get outputs
            boxes, classes, scores, num = get_output_tensors(interpreter)
            if boxes is None or classes is None or scores is None:
                # If model outputs not recognized
                boxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int32)
                scores = np.zeros((0,), dtype=np.float32)
                num = 0

            n = int(num) if num is not None else min(len(scores), len(boxes))
            n = min(n, len(scores), len(boxes), len(classes))

            # Collect detections above threshold
            detections = []
            for i in range(n):
                score = float(scores[i])
                if score < CONFIDENCE_THRESHOLD:
                    continue
                ymin, xmin, ymax, xmax = boxes[i]
                class_id = int(classes[i])
                detections.append((ymin, xmin, ymax, xmax, score, class_id))

            # Draw detections
            frame_out = frame_bgr.copy()
            draw_detections(frame_out, detections, labels, color_cache)

            # Update and draw proxy mAP
            current_map = update_stats_and_compute_map(stats_by_class, detections)
            map_text = f"mAP: {current_map:.3f}"
            cv2.putText(frame_out, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 200, 30), 2, cv2.LINE_AA)

            # Write frame to output
            writer.write(frame_out)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    final_map = float(np.mean([np.mean(v) for v in stats_by_class.values()])) if stats_by_class else 0.0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s ({(frame_count/elapsed) if elapsed>0 else 0:.2f} FPS).")
    print(f"Saved output video to: {OUTPUT_PATH}")
    print(f"Final mAP (proxy): {final_map:.3f}")

if __name__ == "__main__":
    main()