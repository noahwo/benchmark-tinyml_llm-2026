import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# EdgeTPU shared library path for Coral Dev Board
edgetpu_shared_lib = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


# =========================
# Utility Functions
# =========================
def load_labels(path):
    """
    Load labels from a label map file.
    Supports formats:
      - "0 person"
      - "person" (index auto-assigned)
      - Ignores empty lines and lines starting with '#'
    Returns: dict[int, str]
    """
    labels = {}
    try:
        with open(path, 'r') as f:
            idx_auto = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1].strip()
                else:
                    idx = idx_auto
                    name = line
                    idx_auto += 1
                labels[idx] = name
    except Exception as e:
        print(f"Warning: Failed to load labels from {path}: {e}")
    return labels


def make_interpreter(model_path, delegate_path):
    """
    Create a TFLite interpreter with EdgeTPU delegate.
    """
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(delegate_path)]
        )
    except ValueError as e:
        raise RuntimeError(f"Failed to load the EdgeTPU delegate from {delegate_path}: {e}")
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame_bgr, input_h, input_w, input_dtype, input_quant):
    """
    Preprocess frame:
      - Convert BGR to RGB
      - Resize to model input size
      - Quantize/normalize according to input dtype
    Returns: np.ndarray with shape [1, H, W, 3] and proper dtype
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.uint8:
        input_data = resized.astype(np.uint8)
    elif input_dtype == np.float32:
        input_data = (resized.astype(np.float32) / 255.0).astype(np.float32)
    else:
        # Handle other quantized types (e.g., int8) via provided quantization params if available
        scale, zero_point = input_quant if input_quant is not None else (0.0, 0)
        if scale and scale > 0:
            # Assume real-value input in [0,1]; normalize then quantize
            norm = resized.astype(np.float32) / 255.0
            quantized = norm / scale + zero_point
            qmin, qmax = (np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
            input_data = np.clip(np.round(quantized), qmin, qmax).astype(input_dtype)
        else:
            # Fallback: direct cast
            input_data = resized.astype(input_dtype)

    # Add batch dimension
    return np.expand_dims(input_data, axis=0)


def parse_detections(interpreter, output_details, frame_w, frame_h, score_threshold):
    """
    Parse detections from interpreter outputs. Assumes standard TFLite detection output:
      - boxes: [1, N, 4] (ymin, xmin, ymax, xmax) normalized
      - classes: [1, N]
      - scores: [1, N]
      - num_detections: [1]
    Returns: list of dict with keys: bbox (x1,y1,x2,y2), class_id, score
    """
    # Standard order in most TFLite detection models
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num = int(interpreter.get_tensor(output_details[3]['index'])[0]) if len(output_details) > 3 else len(scores)

    detections = []
    for i in range(num):
        score = float(scores[i])
        if score < score_threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        # Convert to absolute coordinates
        x1 = int(max(0, min(frame_w - 1, x_min * frame_w)))
        y1 = int(max(0, min(frame_h - 1, y_min * frame_h)))
        x2 = int(max(0, min(frame_w - 1, x_max * frame_w)))
        y2 = int(max(0, min(frame_h - 1, y_max * frame_h)))
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "class_id": int(classes[i]),
            "score": score
        })
    return detections


def draw_detections(frame_bgr, detections, labels, map_value=None):
    """
    Draw bounding boxes and labels on the frame.
    Optionally overlay running mAP value.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cid = det["class_id"]
        score = det["score"]
        label = labels.get(cid, str(cid))
        # Deterministic color per class
        color = (int(37 * (cid + 1) % 255), int(17 * (cid + 1) % 255), int(29 * (cid + 1) % 255))

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_bgr, caption, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay running mAP value if provided
    if map_value is not None:
        text = f"mAP: {map_value:.3f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame_bgr, (8, 8), (8 + tw + 12, 8 + th + 14), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (14, 8 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr


def compute_running_map(confidences_by_class):
    """
    Compute a running mAP proxy over seen detections.
    Since no ground truth is provided, we approximate AP per class as
    the mean confidence of detections for that class, and mAP as the mean
    of these per-class means.
    Returns: float in [0,1] (approximation).
    """
    per_class_avgs = []
    for _, scores in confidences_by_class.items():
        if scores:
            per_class_avgs.append(float(np.mean(scores)))
    if not per_class_avgs:
        return 0.0
    return float(np.mean(per_class_avgs))


# =========================
# Main Processing Pipeline
# =========================
def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter with EdgeTPU
    interpreter = make_interpreter(model_path, edgetpu_shared_lib)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    in_h, in_w = int(input_details['shape'][1]), int(input_details['shape'][2])
    in_dtype = input_details['dtype']
    in_quant = input_details.get('quantization', (0.0, 0)) if 'quantization' in input_details else (0.0, 0)

    # Video IO setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback FPS

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # For running mAP approximation
    confidences_by_class = {}
    frame_idx = 0
    t0 = time.time()
    last_log = t0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Preprocess
            input_tensor = preprocess_frame(frame, in_h, in_w, in_dtype, in_quant)
            interpreter.set_tensor(input_details['index'], input_tensor)

            # Inference
            inf_start = time.time()
            interpreter.invoke()
            inf_end = time.time()

            # Parse detections
            detections = parse_detections(interpreter, output_details, frame_w, frame_h, confidence_threshold)

            # Update confidences for mAP approximation
            for det in detections:
                cls_id = det["class_id"]
                score = det["score"]
                if cls_id not in confidences_by_class:
                    confidences_by_class[cls_id] = []
                confidences_by_class[cls_id].append(score)

            # Compute running mAP
            running_map = compute_running_map(confidences_by_class)

            # Draw and write frame
            annotated = draw_detections(frame.copy(), detections, labels, map_value=running_map)
            writer.write(annotated)

            frame_idx += 1

            # Optional progress logging every few seconds
            now = time.time()
            if now - last_log > 5.0:
                rt_fps = frame_idx / (now - t0 + 1e-9)
                print(f"[Progress] Frames processed: {frame_idx}, Avg FPS: {rt_fps:.2f}, Running mAP: {running_map:.3f}")
                last_log = now

    finally:
        cap.release()
        writer.release()

    final_map = compute_running_map(confidences_by_class)
    print(f"Processing complete.")
    print(f"Output saved to: {output_path}")
    print(f"Approximate mAP over video: {final_map:.4f}")


if __name__ == "__main__":
    main()