import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# ----------------------------
# Configuration Parameters
# ----------------------------
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# ----------------------------
# Utility Functions
# ----------------------------
def load_labels(path):
    """
    Load labels from a file.
    Supports formats:
    - index label
    - index: label
    - label (implies incremental indexing)
    """
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    auto_index = 0
    for line in lines:
        if ":" in line:
            # index: label
            try:
                idx_str, name = line.split(":", 1)
                idx = int(idx_str.strip())
                labels[idx] = name.strip()
            except Exception:
                labels[auto_index] = line.strip()
                auto_index += 1
        elif " " in line:
            # index label
            parts = line.split()
            if parts[0].isdigit():
                try:
                    idx = int(parts[0])
                    name = " ".join(parts[1:]).strip()
                    labels[idx] = name
                except Exception:
                    labels[auto_index] = line.strip()
                    auto_index += 1
            else:
                labels[auto_index] = line.strip()
                auto_index += 1
        else:
            labels[auto_index] = line.strip()
            auto_index += 1
    return labels

def initialize_interpreter(model_path):
    """
    Initialize TFLite interpreter with EdgeTPU delegate when available.
    """
    try:
        delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(delegate_path)]
        )
        used_delegate = "EdgeTPU"
    except Exception as e:
        # Fallback to CPU if delegate not available
        interpreter = Interpreter(model_path=model_path)
        used_delegate = "CPU"
    interpreter.allocate_tensors()
    return interpreter, used_delegate

def get_input_size(interpreter):
    """
    Get input tensor shape as (height, width).
    """
    input_details = interpreter.get_input_details()[0]
    shape = input_details['shape']
    # Typical shape: [1, height, width, 3]
    if len(shape) == 4:
        if shape[3] == 3:
            return int(shape[1]), int(shape[2])
        elif shape[1] == 3:
            # If channels first (unlikely for TFLite), swap accordingly
            return int(shape[2]), int(shape[3])
    # Fallback
    return 300, 300

def set_input_tensor(interpreter, frame_bgr):
    """
    Preprocess BGR frame to model input and set it to interpreter.
    """
    input_details = interpreter.get_input_details()[0]
    h, w = get_input_size(interpreter)
    # Convert BGR to RGB, resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))
    input_data = resized
    # Handle dtype
    if input_details['dtype'] == np.float32:
        input_data = (input_data.astype(np.float32) - 127.5) / 127.5
    else:
        input_data = input_data.astype(input_details['dtype'])
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details['index'], input_data)

def get_detections(interpreter, im_w, im_h, score_threshold):
    """
    Get detections from interpreter outputs, convert to image coordinates.
    Returns list of dicts with keys: bbox, score, class_id
    bbox format: (xmin, ymin, xmax, ymax) in absolute pixel coords.
    """
    output_details = interpreter.get_output_details()
    # Common order for SSD MobileNet V2/EdgeTPU: boxes, classes, scores, count
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    boxes, classes, scores, count = None, None, None, None

    # Attempt to infer outputs by shapes
    for out in outputs:
        arr = np.squeeze(out)
        if arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = arr
        elif arr.ndim == 1 and arr.size <= 100 and arr.dtype in [np.float32, np.float64]:
            # Heuristic: scores often 1D float
            # We'll assign later once we find the matching shape
            pass
        elif arr.ndim == 1 and arr.size <= 100 and np.issubdtype(arr.dtype, np.integer):
            # classes likely 1D int
            pass

    # Fallback to standard indexing if heuristic not decisive
    if boxes is None and len(outputs) >= 3:
        try:
            boxes = np.squeeze(outputs[0])
            classes = np.squeeze(outputs[1]).astype(np.int32)
            scores = np.squeeze(outputs[2])
            if len(outputs) >= 4:
                count = int(np.squeeze(outputs[3]))
        except Exception:
            pass

    # If still not mapped, try assign by shapes explicitly
    if boxes is None:
        for out in outputs:
            arr = np.squeeze(out)
            if arr.ndim == 2 and arr.shape[-1] == 4:
                boxes = arr
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
                if scores is None or arr.size > scores.size:
                    scores = arr
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                classes = arr.astype(np.int32)
    if count is None:
        # If count tensor not provided, use min length of arrays
        if boxes is not None:
            count = boxes.shape[0]
        elif scores is not None:
            count = scores.shape[0]
        else:
            count = 0

    detections = []
    if boxes is None or scores is None or classes is None:
        return detections

    count = min(count, len(scores), len(classes), len(boxes))
    for i in range(count):
        score = float(scores[i])
        if score < score_threshold:
            continue
        # Boxes are typically [ymin, xmin, ymax, xmax] normalized [0,1]
        y_min, x_min, y_max, x_max = boxes[i]
        x_min = max(0, min(int(x_min * im_w), im_w - 1))
        x_max = max(0, min(int(x_max * im_w), im_w - 1))
        y_min = max(0, min(int(y_min * im_h), im_h - 1))
        y_max = max(0, min(int(y_max * im_h), im_h - 1))
        if x_max <= x_min or y_max <= y_min:
            continue
        detections.append({
            "bbox": (x_min, y_min, x_max, y_max),
            "score": score,
            "class_id": int(classes[i])
        })
    return detections

def draw_detections(frame, detections, labels, map_score=None):
    """
    Draw detection bounding boxes and labels on the frame.
    Optionally draw mAP score on the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        score = det["score"]
        label = labels.get(class_id, f"id:{class_id}")
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        caption = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw mAP (proxy) on the top-left corner
    if map_score is not None:
        text = f"mAP: {map_score:.3f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (5, 5), (5 + tw + 8, 5 + th + baseline + 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (9, 9 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

def update_map_proxy(accumulator, detections):
    """
    Update a proxy mAP accumulator using detection confidences.
    This is NOT a true mAP (requires ground truth); it's a proxy based on average confidence per class.
    accumulator: dict[class_id] -> list of confidences
    """
    for det in detections:
        cls = det["class_id"]
        sc = float(det["score"])
        if cls not in accumulator:
            accumulator[cls] = []
        accumulator[cls].append(sc)

def compute_map_proxy(accumulator):
    """
    Compute a proxy mAP as the mean of average confidences across classes.
    """
    if not accumulator:
        return 0.0
    ap_values = []
    for cls, scores in accumulator.items():
        if len(scores) == 0:
            continue
        ap_values.append(float(np.mean(scores)))
    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))

# ----------------------------
# Main Application
# ----------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU if available
    interpreter, device = initialize_interpreter(MODEL_PATH)
    input_h, input_w = get_input_size(interpreter)

    # Open video input
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Unable to open input video: {INPUT_PATH}")
        return

    im_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (im_w, im_h))
    if not writer.isOpened():
        print(f"ERROR: Unable to open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH if os.path.exists(LABEL_PATH) else '(not found)'}")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Device: {device}")
    print(f"Input tensor size (HxW): {input_h}x{input_w}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # Proxy mAP accumulator: class_id -> list of confidences
    map_accumulator = {}

    # Processing loop
    frame_index = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_inf = time.time()
            # Preprocess and run inference
            set_input_tensor(interpreter, frame)
            interpreter.invoke()
            # Postprocess detections
            detections = get_detections(interpreter, im_w, im_h, CONFIDENCE_THRESHOLD)
            inf_time_ms = (time.time() - start_inf) * 1000.0

            # Update mAP proxy and compute current value
            update_map_proxy(map_accumulator, detections)
            map_score = compute_map_proxy(map_accumulator)

            # Draw results
            draw_detections(frame, detections, labels, map_score=map_score)

            # Optionally overlay inference time/FPS
            fps_text = f"{1000.0/inf_time_ms:.2f} FPS" if inf_time_ms > 0 else "FPS: inf"
            cv2.putText(frame, fps_text, (5, im_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Detections: {len(detections)}", (im_w - 170, im_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

            # Write frame to output
            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    avg_fps = frame_index / total_time if total_time > 0 else 0.0
    final_map = compute_map_proxy(map_accumulator)

    print(f"Processed frames: {frame_index}")
    print(f"Total time: {total_time:.2f}s, Average FPS: {avg_fps:.2f}")
    print(f"Final mAP (proxy, no GT): {final_map:.4f}")
    print(f"Saved output video to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()