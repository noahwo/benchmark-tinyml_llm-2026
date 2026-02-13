import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# ==============================
# Configuration Parameters
# ==============================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


# ==============================
# Utilities
# ==============================
def load_labels(path):
    """
    Loads labels from a label map file.
    Supports:
      - one label per line (index inferred from line number)
      - "index label" (space or colon separated)
    """
    labels = {}
    try:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print("Failed to read label file:", e)
        return labels

    for idx, line in enumerate(lines):
        # Try split by common separators
        label = None
        key = None
        if ':' in line:
            parts = line.split(':', 1)
            try:
                key = int(parts[0].strip())
                label = parts[1].strip()
            except ValueError:
                pass
        elif ' ' in line:
            parts = line.split(' ', 1)
            try:
                key = int(parts[0].strip())
                label = parts[1].strip()
            except ValueError:
                pass

        if key is None or label is None:
            # Fallback: use line index as id
            key = idx
            label = line

        labels[key] = label
    return labels


def make_interpreter(model_path, edgetpu_lib_path):
    """
    Creates and returns a TFLite interpreter loaded with EdgeTPU delegate.
    """
    try:
        delegate = load_delegate(edgetpu_lib_path)
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        raise RuntimeError(f"Failed to create TFLite interpreter with EdgeTPU delegate: {e}")


def preprocess_frame(frame_bgr, input_details):
    """
    Preprocess frame to match model input requirements.
    - Resize to expected input size
    - Convert BGR to RGB
    - Type conversion based on model input dtype
    """
    ih, iw = input_details['shape'][1], input_details['shape'][2]
    frame_resized = cv2.resize(frame_bgr, (iw, ih))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_dtype = input_details['dtype']

    if input_dtype == np.float32:
        input_data = (frame_rgb.astype(np.float32) / 255.0).reshape(1, ih, iw, 3)
    else:
        # Assume quantized uint8 input
        input_data = frame_rgb.astype(np.uint8).reshape(1, ih, iw, 3)

    return input_data


def get_detections(interpreter, frame_w, frame_h, confidence_threshold):
    """
    Extract detections from model outputs, convert to pixel coordinates, and filter by confidence threshold.
    Returns a list of dicts: { 'bbox': (x1, y1, x2, y2), 'score': float, 'class_id': int }
    """
    output_details = interpreter.get_output_details()
    # Typical order for TFLite SSD: boxes, classes, scores, count
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [N,4] in ymin,xmin,ymax,xmax normalized
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # [N]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [N]
        count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    except Exception:
        # Attempt to find tensors by shape heuristics if order differs
        boxes = None
        classes = None
        scores = None
        count = None
        for od in output_details:
            tensor = interpreter.get_tensor(od['index'])
            shp = tensor.shape
            if len(shp) == 3 and shp[-1] == 4:
                boxes = tensor[0]
            elif len(shp) == 2:
                # could be classes or scores
                if tensor.dtype == np.float32:
                    # Heuristic: keep higher variance as scores if both float32
                    if scores is None:
                        scores = tensor[0]
                    else:
                        # choose the one whose values are in [0,1] as scores
                        t0 = np.clip(tensor[0], 0.0, 1.0)
                        if np.allclose(t0, tensor[0], atol=1e-5):
                            scores = tensor[0]
                        else:
                            classes = tensor[0]
                else:
                    classes = tensor[0]
            elif len(shp) == 1 and shp[0] == 1:
                count = int(tensor[0])

        if boxes is None or classes is None or scores is None or count is None:
            raise RuntimeError("Unable to parse model outputs for detection.")

    detections = []
    n = min(len(scores), count)
    for i in range(n):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        cls_id = int(classes[i]) if not np.isnan(classes[i]) else -1
        ymin, xmin, ymax, xmax = boxes[i]
        # Clamp and scale to pixel coordinates
        xmin = max(0.0, min(1.0, float(xmin)))
        xmax = max(0.0, min(1.0, float(xmax)))
        ymin = max(0.0, min(1.0, float(ymin)))
        ymax = max(0.0, min(1.0, float(ymax)))

        x1 = int(xmin * frame_w)
        y1 = int(ymin * frame_h)
        x2 = int(xmax * frame_w)
        y2 = int(ymax * frame_h)

        detections.append({
            'bbox': (x1, y1, x2, y2),
            'score': score,
            'class_id': cls_id
        })
    return detections


def draw_detections(frame, detections, labels, map_value=None):
    """
    Draws bounding boxes and labels on the frame.
    Optionally overlay an mAP value (or proxy) at the top-left.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        cls_id = det['class_id']
        label = labels.get(cls_id, f"id:{cls_id}")
        caption = f"{label} {score:.2f}"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Draw label background
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 200, 0), -1)
        # Draw label text
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Overlay mAP (or proxy) at the top-left corner
    if map_value is not None:
        text = f"mAP: {map_value:.3f}"
    else:
        text = "mAP: N/A"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 230), 2)


def compute_proxy_map(all_scores):
    """
    Computes a proxy for mAP in absence of ground truth.
    This proxy computes mean precision across thresholds [0.5, 0.95] where
    precision at threshold t is (#detections with score >= t) / (total detections).
    Note: This is NOT true mAP. It is a placeholder metric.
    """
    if not all_scores:
        return 0.0
    scores = np.array(all_scores, dtype=np.float32)
    thresholds = np.arange(0.5, 1.0, 0.05)
    precisions = []
    total = float(len(scores))
    for t in thresholds:
        precisions.append(float(np.sum(scores >= t)) / total)
    return float(np.mean(precisions)) if precisions else 0.0


# ==============================
# Main pipeline
# ==============================
def main():
    # Prepare output directory
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Setup TFLite + EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_LIB)
    input_details = interpreter.get_input_details()[0]

    # Setup video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # Fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # For proxy mAP computation
    all_detection_scores = []

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Preprocess
            input_data = preprocess_frame(frame, input_details)
            interpreter.set_tensor(input_details['index'], input_data)

            # Inference
            start_time = time.time()
            interpreter.invoke()
            _ = (time.time() - start_time) * 1000.0  # inference time in ms (not displayed but measured)

            # Postprocess detections
            detections = get_detections(interpreter, frame_w=width, frame_h=height, confidence_threshold=CONFIDENCE_THRESHOLD)

            # Update scores for proxy mAP
            for d in detections:
                all_detection_scores.append(d['score'])

            proxy_map = compute_proxy_map(all_detection_scores)

            # Draw results and overlay proxy mAP
            draw_detections(frame, detections, labels, map_value=proxy_map)

            # Write frame
            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    print(f"Processing complete. Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()