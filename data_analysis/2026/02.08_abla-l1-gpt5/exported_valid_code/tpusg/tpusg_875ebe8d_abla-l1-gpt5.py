import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# Utility functions
# =========================
def load_labels(path):
    """
    Load labels from a label map file.
    Supports either:
    - one label per line (index inferred by line order starting at 0)
    - or "index label" per line (space separated), where index is an integer id.
    """
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    # Try to parse "index label" format; fallback to line order.
    parsed_any_index = False
    for i, line in enumerate(lines):
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
            parsed_any_index = True
        else:
            # Will handle later if not index-based
            pass
    if not parsed_any_index:
        # Fallback: one label per line, index by order.
        for i, line in enumerate(lines):
            if line:
                labels[i] = line
    return labels

def make_interpreter(model_path):
    """
    Create TFLite interpreter with EdgeTPU delegate.
    """
    try:
        delegate = load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
    except Exception:
        # Fallback to CPU (should not happen on Coral Dev Board if EdgeTPU is available)
        interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame_bgr, input_size, input_dtype):
    """
    Convert a BGR frame to model's input tensor format.
    - Resizes to input_size (width, height)
    - Converts to RGB
    - Scales if model expects float32, else keeps uint8 range
    Returns tensor with shape [1, height, width, 3] and dtype input_dtype.
    """
    in_w, in_h = input_size
    # Convert BGR->RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        # Normalize to [0,1] as a safe default for float models
        resized = resized.astype(np.float32) / 255.0
    else:
        resized = resized.astype(np.uint8)
    # Add batch dimension
    input_tensor = np.expand_dims(resized, axis=0).astype(input_dtype)
    return input_tensor

def get_output_tensors(interpreter):
    """
    Extract detection outputs from the interpreter.
    Assumes SSD-style output:
      - boxes: [1, N, 4] with (ymin, xmin, ymax, xmax) normalized [0,1]
      - classes: [1, N]
      - scores: [1, N]
      - count: [1] (int)
    Returns (boxes, classes, scores, count)
    """
    output_details = interpreter.get_output_details()
    # Heuristic to identify outputs by shapes/dtypes
    boxes = classes = scores = num = None
    for od in output_details:
        out = interpreter.get_tensor(od['index'])
        shp = out.shape
        if len(shp) == 3 and shp[-1] == 4:
            boxes = out[0]
        elif len(shp) == 2:
            # Could be classes or scores
            if out.dtype == np.float32:
                scores = out[0]
            else:
                # Some models output classes as float; cast later
                classes = out[0]
        elif len(shp) == 1 and shp[0] == 1:
            num = int(out[0])
    # If classes is float array, cast to int
    if classes is not None and classes.dtype != np.int32:
        classes = classes.astype(np.int32)
    # Safety fallbacks
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.int32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)
    if num is None:
        num = min(len(scores), len(classes), len(boxes))
    return boxes, classes, scores, num

def draw_detections(frame_bgr, detections, labels, map_text, font_scale=0.6, thickness=2):
    """
    Draw bounding boxes and labels on the frame.
    detections: list of dicts with keys: xmin, ymin, xmax, ymax, class_id, score
    map_text: string to display for mAP
    """
    h, w = frame_bgr.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det['xmin'], det['ymin'], det['xmax'], det['ymax']
        class_id = det['class_id']
        score = det['score']
        label = labels.get(class_id, str(class_id))
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}: {score:.2f}"
        # Put label background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y_text = max(y1 - 10, th + 4)
        cv2.rectangle(frame_bgr, (x1, y_text - th - 4), (x1 + tw + 2, y_text + baseline - 2), (0, 0, 0), -1)
        cv2.putText(frame_bgr, caption, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # Draw mAP text (N/A since ground truth is not provided)
    map_caption = f"mAP: {map_text}"
    (tw, th), baseline = cv2.getTextSize(map_caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame_bgr, (5, 5), (10 + tw, 10 + th + baseline), (0, 0, 0), -1)
    cv2.putText(frame_bgr, map_caption, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame_bgr

# =========================
# Main application
# =========================
def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize interpreter
    interpreter = make_interpreter(model_path)
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    # Input tensor shape [1, height, width, 3]
    _, in_h, in_w, in_c = input_details['shape']
    input_dtype = input_details['dtype']

    # Open video input
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open input video: {input_path}")
        return

    # Prepare video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not out_writer.isOpened():
        print(f"ERROR: Could not open output video for write: {output_path}")
        cap.release()
        return

    # Processing loop
    frame_count = 0
    t0 = time.time()
    # Since ground-truth annotations are not provided, we display mAP as N/A.
    map_text = "N/A"

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            input_tensor = preprocess_frame(frame_bgr, (in_w, in_h), input_dtype)
            interpreter.set_tensor(input_index, input_tensor)

            # Inference
            interpreter.invoke()

            # Postprocess
            boxes, classes, scores, num = get_output_tensors(interpreter)
            num = min(num, len(scores), len(classes), len(boxes))
            h, w = frame_bgr.shape[:2]
            detections = []
            for i in range(num):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                cls_id = int(classes[i])
                # boxes are [ymin, xmin, ymax, xmax] normalized in [0,1]
                ymin = max(0.0, min(1.0, float(boxes[i][0])))
                xmin = max(0.0, min(1.0, float(boxes[i][1])))
                ymax = max(0.0, min(1.0, float(boxes[i][2])))
                xmax = max(0.0, min(1.0, float(boxes[i][3])))
                x1 = int(xmin * w)
                y1 = int(ymin * h)
                x2 = int(xmax * w)
                y2 = int(ymax * h)
                # Clamp to image bounds
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({
                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                    'class_id': cls_id, 'score': score
                })

            # Draw and annotate
            annotated = draw_detections(frame_bgr, detections, labels, map_text)
            out_writer.write(annotated)

    finally:
        cap.release()
        out_writer.release()

    elapsed = time.time() - t0
    if elapsed > 0:
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} FPS)")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()