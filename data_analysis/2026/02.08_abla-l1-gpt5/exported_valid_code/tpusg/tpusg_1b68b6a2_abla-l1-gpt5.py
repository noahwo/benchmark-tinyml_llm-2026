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

# =========================
# Utility Functions
# =========================
def load_labels(path):
    """Load label map from file. Supports 'index label' or plain lines."""
    labels = {}
    if not os.path.isfile(path):
        print(f"Warning: Label file not found at {path}. Using empty labels.")
        return labels
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for i, line in enumerate(lines):
        # Try "index label" format
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
        else:
            # Fallback: sequential labels
            labels[i] = line
    return labels

def make_interpreter_tpu(model_path_):
    """Create a TFLite interpreter with EdgeTPU delegate."""
    delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    interpreter = Interpreter(
        model_path=model_path_,
        experimental_delegates=[load_delegate(delegate_path)]
    )
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    """Return model input details and size."""
    input_details = interpreter.get_input_details()
    in_idx = input_details[0]['index']
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    dtype = input_details[0]['dtype']
    quant = input_details[0].get('quantization', (0.0, 0))  # (scale, zero_point)
    return in_idx, (width, height), dtype, quant

def set_input(interpreter, in_index, frame_bgr, input_size, dtype, quant):
    """Preprocess and set input tensor for the interpreter."""
    width, height = input_size
    # Convert BGR -> RGB and resize to model input
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    input_tensor = np.expand_dims(resized, axis=0)

    if dtype == np.uint8:
        # For quantized models (EdgeTPU), input is typically uint8 [0, 255]
        input_tensor = input_tensor.astype(np.uint8)
    else:
        # For float models, normalize to [0,1]
        input_tensor = (input_tensor.astype(np.float32) / 255.0).astype(np.float32)

    interpreter.set_tensor(in_index, input_tensor)

def parse_detections(interpreter, score_threshold, frame_shape):
    """Parse detection outputs from the interpreter and return valid detections."""
    output_details = interpreter.get_output_details()

    # Attempt typical SSD output ordering
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [N, 4]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # [N]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [N]
        count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    except Exception:
        # Fallback: find tensors by shapes
        boxes = classes = scores = None
        count = 0
        for od in output_details:
            data = interpreter.get_tensor(od['index'])
            shape = data.shape
            if len(shape) == 3 and shape[-1] == 4:
                boxes = data[0]
            elif len(shape) == 2 and shape[0] == 1:
                # Could be classes or scores
                if data.dtype in (np.float32, np.float64):
                    if np.max(data[0]) <= 1.0:
                        scores = data[0]
                    else:
                        classes = data[0]
            elif len(shape) == 1 and shape[0] == 1:
                count = int(data[0])
        if boxes is None or classes is None or scores is None:
            return []

    h, w = frame_shape[:2]
    detections = []
    for i in range(count):
        score = float(scores[i])
        if score < score_threshold:
            continue
        # boxes are in y_min, x_min, y_max, x_max (normalized)
        y_min, x_min, y_max, x_max = boxes[i]
        x_min_abs = max(0, int(x_min * w))
        x_max_abs = min(w - 1, int(x_max * w))
        y_min_abs = max(0, int(y_min * h))
        y_max_abs = min(h - 1, int(y_max * h))
        cls_id = int(classes[i]) if not np.isnan(classes[i]) else -1
        detections.append({
            "bbox": (x_min_abs, y_min_abs, x_max_abs, y_max_abs),
            "score": score,
            "class_id": cls_id
        })
    return detections

def draw_detections(frame, detections, labels, map_text="mAP: N/A"):
    """Draw rectangles and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cls_id = det["class_id"]
        label = labels.get(cls_id, f"id:{cls_id}")
        caption = f"{label} {score:.2f}"
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # Draw background for text
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw mAP text on top-left
    cv2.putText(frame, map_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 255), 2, cv2.LINE_AA)
    return frame

def ensure_dir_for_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# =========================
# Main Processing
# =========================
def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize interpreter with EdgeTPU delegate
    interpreter = make_interpreter_tpu(model_path)
    in_index, input_size, input_dtype, input_quant = get_input_details(interpreter)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    # Prepare output video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # For perf/debug
    frame_count = 0
    inference_times = []

    # Since no ground-truth annotations are provided, mAP cannot be computed.
    # We'll annotate frames with "mAP: N/A".
    map_text = "mAP: N/A"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess and set input
            set_input(interpreter, in_index, frame, input_size, input_dtype, input_quant)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            inference_times.append((t1 - t0) * 1000.0)  # ms

            # Parse detections
            detections = parse_detections(interpreter, confidence_threshold, frame.shape)

            # Draw and write frame
            annotated = draw_detections(frame, detections, labels, map_text=map_text)
            writer.write(annotated)

    finally:
        cap.release()
        writer.release()

    # Summary
    if inference_times:
        avg_ms = sum(inference_times) / len(inference_times)
        print(f"Processed {frame_count} frames.")
        print(f"Average inference time: {avg_ms:.2f} ms per frame")
        print(f"Output saved to: {output_path}")
        print("Note: mAP not computed due to missing ground-truth annotations.")
    else:
        print("No frames processed. Please check the input video path.")

if __name__ == "__main__":
    main()