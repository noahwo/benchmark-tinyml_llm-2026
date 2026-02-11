import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# ---------------------------
# Configuration parameters
# ---------------------------
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5

EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# ---------------------------
# Helpers
# ---------------------------
def load_labels(label_path):
    labels = {}
    if not os.path.isfile(label_path):
        print("Warning: label file not found; classes will be numeric IDs.")
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Support two common formats:
            # 1) "0 person"
            # 2) "person"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
            else:
                labels[idx] = line
    return labels

def make_interpreter(model_path):
    # Ensure EdgeTPU delegate is used
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(EDGETPU_LIB)]
        )
    except ValueError as e:
        raise RuntimeError(f"Failed to load EdgeTPU delegate: {e}")
    interpreter.allocate_tensors()
    return interpreter

def preprocess(frame_bgr, input_shape, input_dtype):
    # Convert to RGB and resize to model input size
    _, in_h, in_w, _ = input_shape
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.uint8:
        input_data = resized
    else:
        # Assume float model expects [0,1]
        input_data = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def get_output_tensors(interpreter):
    output_details = interpreter.get_output_details()
    outputs = {}
    # Try to map by tensor name when available
    for od in output_details:
        name = od.get('name', '')
        arr = interpreter.get_tensor(od['index'])
        lname = name.lower()
        if 'box' in lname:
            outputs['boxes'] = arr
        elif 'score' in lname or 'confidence' in lname:
            outputs['scores'] = arr
        elif 'class' in lname:
            outputs['classes'] = arr
        elif 'count' in lname or 'num' in lname:
            outputs['count'] = arr

    # Fallback: deduce by shapes if names failed
    if not outputs.get('boxes') or not outputs.get('scores') or not outputs.get('classes'):
        for od in output_details:
            arr = interpreter.get_tensor(od['index'])
            shape = arr.shape
            if len(shape) == 3 and shape[0] == 1 and shape[2] == 4:
                outputs['boxes'] = arr
            elif len(shape) == 2 and shape[0] == 1:
                # Could be scores or classes
                if arr.dtype in (np.float32, np.float16):
                    # Scores are typically float
                    # Heuristic: scores between 0 and 1
                    if np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
                        outputs.setdefault('scores', arr)
                    else:
                        # Some models may use floats for classes too, handle later
                        outputs.setdefault('classes', arr)
                else:
                    outputs.setdefault('classes', arr)
            elif len(shape) == 1 and shape[0] == 1:
                outputs['count'] = arr

    return outputs

def parse_detections(outputs, frame_w, frame_h, threshold, labels_dict):
    # Expected shapes: boxes [1, N, 4], classes [1, N], scores [1, N], count [1]
    boxes = outputs.get('boxes')
    scores = outputs.get('scores')
    classes = outputs.get('classes')
    count = outputs.get('count')

    if boxes is None or scores is None or classes is None:
        return []

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    if count is not None:
        num = int(np.squeeze(count))
        # Some models may provide more entries than 'num'
        boxes = boxes[:num]
        scores = scores[:num]
        classes = classes[:num]

    # Classes may be float in some models; cast to int safely
    classes = classes.astype(np.int32)

    detections = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        # Boxes usually in [ymin, xmin, ymax, xmax] normalized to [0,1]
        ymin, xmin, ymax, xmax = boxes[i]
        ymin = max(0.0, min(1.0, float(ymin)))
        xmin = max(0.0, min(1.0, float(xmin)))
        ymax = max(0.0, min(1.0, float(ymax)))
        xmax = max(0.0, min(1.0, float(xmax)))

        x1 = int(xmin * frame_w)
        y1 = int(ymin * frame_h)
        x2 = int(xmax * frame_w)
        y2 = int(ymax * frame_h)

        class_id = int(classes[i])
        label = labels_dict.get(class_id, str(class_id))
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'score': score,
            'class_id': class_id,
            'label': label
        })
    return detections

def draw_detections(frame_bgr, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        score = det['score']
        # Draw rectangle
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # Label with background
        caption = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - 5)
        cv2.rectangle(frame_bgr, (x1, y_text - th - baseline), (x1 + tw + 2, y_text + baseline), (0, 200, 0), -1)
        cv2.putText(frame_bgr, caption, (x1 + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def overlay_map_info(frame_bgr, map_value):
    text = f"mAP: {map_value}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (5, 5), (5 + tw + 10, 5 + th + 10), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

def main():
    # 1) Setup: Interpreter, labels, video I/O
    labels = load_labels(LABEL_PATH)
    interpreter = make_interpreter(MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {OUTPUT_PATH}")

    # 2-3) Process frames: Preprocess -> Inference -> Postprocess
    frame_count = 0
    t_infer_sum = 0.0

    # mAP requires ground truth; since it's not provided, we'll report N/A.
    # You can integrate ground-truth annotations and compute true mAP if available.
    computed_map_display = "N/A (no ground truth)"

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            input_data = preprocess(frame_bgr, input_shape, input_dtype)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            t_infer_sum += (t1 - t0)

            # Get outputs
            outputs = get_output_tensors(interpreter)
            detections = parse_detections(outputs, width, height, CONF_THRESHOLD, labels)

            # 4) Draw results and overlay mAP
            draw_detections(frame_bgr, detections)
            overlay_map_info(frame_bgr, computed_map_display)

            writer.write(frame_bgr)
            frame_count += 1

    finally:
        cap.release()
        writer.release()

    if frame_count > 0:
        avg_infer_ms = (t_infer_sum / frame_count) * 1000.0
        print(f"Processed {frame_count} frames")
        print(f"Average inference time: {avg_infer_ms:.2f} ms/frame")
        print(f"Output saved to: {OUTPUT_PATH}")
    else:
        print("No frames processed; please check the input video.")

if __name__ == "__main__":
    main()