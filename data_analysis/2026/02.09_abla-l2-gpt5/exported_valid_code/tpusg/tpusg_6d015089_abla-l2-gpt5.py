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
    """Load label map from file. Supports:
       - 'index label' per line, or
       - single label per line (index inferred from line number).
    """
    labels = {}
    try:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0])
                name = " ".join(parts[1:])
            else:
                idx = i
                name = line
            labels[idx] = name
    except Exception as e:
        print(f"Warning: Failed to load labels from {path}: {e}")
        labels = {}
    return labels

def make_interpreter_tpu(model_path, edgetpu_lib='/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'):
    """Create TFLite interpreter with EdgeTPU delegate."""
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(edgetpu_lib)]
        )
        print("EdgeTPU delegate loaded successfully.")
    except Exception as e:
        # If delegate fails, raise error, as guideline requests TPU usage.
        raise RuntimeError(f"Failed to load EdgeTPU delegate from {edgetpu_lib}: {e}")
    return interpreter

def preprocess_frame_bgr(frame_bgr, input_details):
    """Resize and format frame to model input shape/dtype."""
    _, in_h, in_w, in_c = input_details[0]['shape']
    dtype = input_details[0]['dtype']
    # Convert BGR->RGB and resize
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)
    if dtype == np.float32:
        input_data = (input_data.astype(np.float32) / 255.0)
    elif dtype == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        # Default to original dtype casting
        input_data = input_data.astype(dtype)
    return input_data

def parse_detections(interpreter, output_details):
    """Parse model outputs assuming SSD-style outputs:
       - boxes: [1, num, 4] (ymin, xmin, ymax, xmax), normalized [0,1]
       - classes: [1, num]
       - scores: [1, num]
       - num_detections: [1]
    """
    # Attempt to infer by common ordering
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        if len(output_details) > 3:
            num = int(interpreter.get_tensor(output_details[3]['index'])[0])
        else:
            num = len(scores)
    except Exception:
        # Fallback: search by shapes
        boxes = classes = scores = None
        num = None
        for od in output_details:
            out = interpreter.get_tensor(od['index'])
            shp = out.shape
            if len(shp) == 3 and shp[-1] == 4:
                boxes = out[0]
            elif len(shp) == 2 and shp[0] == 1:
                # Could be classes or scores
                arr = out[0]
                # Heuristic: scores are floats in [0,1]; classes often near small ints or floats of ints
                if arr.dtype.kind == 'f' and np.all((arr >= 0.0) & (arr <= 1.0)):
                    scores = arr
                else:
                    classes = arr.astype(np.int32)
            elif len(shp) == 1 and shp[0] == 1:
                num = int(out[0])
        if num is None and scores is not None:
            num = len(scores)
        if boxes is None or classes is None or scores is None:
            raise RuntimeError("Unable to parse detection outputs from the model.")
    return boxes, classes, scores, num

def draw_detections(frame, detections, labels, approx_map):
    """Draw bounding boxes, labels, and mAP on the frame."""
    h, w = frame.shape[:2]
    for det in detections:
        ymin, xmin, ymax, xmax, cls_id, score = det
        # Scale to pixel coordinates
        left = max(0, int(xmin * w))
        top = max(0, int(ymin * h))
        right = min(w - 1, int(xmax * w))
        bottom = min(h - 1, int(ymax * h))
        # Choose a color based on class id for consistency
        np.random.seed(cls_id + 12345)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        # Label text
        label_text = labels.get(cls_id, str(cls_id))
        caption = f"{label_text} {score:.2f}"
        # Put text background
        (txt_w, txt_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (left, max(0, top - txt_h - baseline)), (left + txt_w, top), color, thickness=-1)
        cv2.putText(frame, caption, (left, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Draw approximate mAP on the top-left
    cv2.putText(frame, f"mAP (approx.): {approx_map:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)

def compute_running_map_approx(per_class_sum, per_class_count):
    """Approximate mAP as the mean of per-class average confidence over observed classes.
       Note: True mAP requires ground-truth and IoU matching; this is a proxy metric.
    """
    aps = []
    for cls_id, cnt in per_class_count.items():
        if cnt > 0:
            aps.append(per_class_sum[cls_id] / float(cnt))
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

# =========================
# Main Pipeline
# =========================
def main():
    # Validate input
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.isfile(label_path):
        print(f"Warning: Label file not found at {label_path}. Proceeding without labels.")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found at {input_path}")

    # Load labels
    labels = load_labels(label_path)

    # Initialize interpreter with EdgeTPU
    interpreter = make_interpreter_tpu(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize video I/O
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0  # default fallback

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer at {output_path}")

    # Stats for approximate mAP
    per_class_sum = {}    # cls_id -> sum(confidence)
    per_class_count = {}  # cls_id -> count
    total_frames = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            # Preprocess
            input_data = preprocess_frame_bgr(frame, input_details)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Inference
            interpreter.invoke()

            # Parse detections
            boxes, classes, scores, num = parse_detections(interpreter, output_details)

            # Collect filtered detections
            filtered = []
            for i in range(num):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                cls_id = int(classes[i])
                ymin, xmin, ymax, xmax = boxes[i]
                filtered.append((ymin, xmin, ymax, xmax, cls_id, score))
                # Update stats
                per_class_sum[cls_id] = per_class_sum.get(cls_id, 0.0) + score
                per_class_count[cls_id] = per_class_count.get(cls_id, 0) + 1

            # Compute approximate mAP
            approx_map = compute_running_map_approx(per_class_sum, per_class_count)

            # Draw detections and mAP
            draw_detections(frame, filtered, labels, approx_map)

            # Write frame
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    approx_map = compute_running_map_approx(per_class_sum, per_class_count)
    print("Processing complete.")
    print(f"Frames processed: {total_frames}")
    print(f"Elapsed time: {elapsed:.2f}s, FPS: { (total_frames / elapsed) if elapsed > 0 else 0.0:.2f}")
    print(f"Approximate mAP (confidence-based proxy): {approx_map:.4f}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()