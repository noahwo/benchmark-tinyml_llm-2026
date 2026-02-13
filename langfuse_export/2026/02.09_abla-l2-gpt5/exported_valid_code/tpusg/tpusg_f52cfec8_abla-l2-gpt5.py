import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =======================
# CONFIGURATION PARAMETERS
# =======================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONF_THRESHOLD = 0.5  # confidence threshold for drawing boxes

# Optional: limit processed frames to avoid long runs in constrained environments.
# On Coral Dev Board, this is typically unset.
MAX_FRAMES = int(os.getenv("TFLITE_MAX_FRAMES", "0"))

# =======================
# HELPER FUNCTIONS
# =======================
def load_labels(path):
    """
    Loads label map from a text file.
    Supports formats:
      - "0: person"
      - "0 person"
      - plain list (index = line number)
    """
    labels = {}
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            key = None
            name = None
            if ':' in line:
                # "id: name"
                parts = line.split(':', 1)
                if parts[0].strip().isdigit():
                    key = int(parts[0].strip())
                    name = parts[1].strip()
            elif ' ' in line[:4]:
                # "id name" (id then name)
                head = line.split(None, 1)
                if head[0].isdigit():
                    key = int(head[0])
                    name = head[1].strip()
            if key is None or name is None:
                # Fallback: plain list
                key = idx
                name = line
            labels[key] = name
    return labels

def make_interpreter(model_path):
    """
    Creates a TFLite interpreter with EdgeTPU delegate.
    """
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    shape = input_details['shape']
    # Expect [1, height, width, channels]
    height, width = int(shape[1]), int(shape[2])
    dtype = input_details['dtype']
    quant = input_details.get('quantization', (0.0, 0))
    return input_details['index'], height, width, dtype, quant

def set_input(interpreter, input_index, frame_bgr, in_h, in_w, in_dtype, in_quant):
    """
    Preprocess frame and set input tensor.
    Handles RGB conversion, resize, dtype and quantization if needed.
    """
    # Convert BGR (OpenCV) to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(rgb, (in_w, in_h))
    if in_dtype == np.float32:
        # Normalize to [0, 1]
        input_data = resized.astype(np.float32) / 255.0
    else:
        # Assume uint8 (quantized)
        input_data = resized.astype(np.uint8)
        # If per-tensor quantization is provided and dtype is uint8, we typically pass raw uint8.
        # (Most EdgeTPU detection models expect uint8 input without manual zero_point/scale here.)
    # Set input tensor
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_index, input_data)

def _dequantize_if_needed(tensor, detail):
    """
    Dequantize tensor if the output is quantized. Returns float32 array.
    """
    if detail['dtype'] == np.uint8:
        scale, zero_point = detail.get('quantization', (0.0, 0))
        if scale and scale > 0:
            return scale * (tensor.astype(np.float32) - zero_point)
        else:
            return tensor.astype(np.float32)
    else:
        return tensor.astype(np.float32)

def get_detections(interpreter):
    """
    Retrieves detection outputs: boxes, classes, scores, num.
    Tries to infer outputs robustly.
    Returns:
      boxes: [N, 4] in normalized coordinates (ymin, xmin, ymax, xmax)
      classes: [N] integers
      scores: [N] floats in [0,1]
      count: integer N (number of valid detections to consider)
    """
    output_details = interpreter.get_output_details()
    outputs = []
    for d in output_details:
        arr = interpreter.get_tensor(d['index'])
        arr = _dequantize_if_needed(arr, d)
        outputs.append((d, arr))

    # Attempt robust identification of outputs
    boxes = None
    classes = None
    scores = None
    num = None

    # Identify boxes: last dim == 4
    for d, arr in outputs:
        if arr.ndim >= 2 and arr.shape[-1] == 4:
            boxes = arr[0] if arr.ndim == 3 else arr
            break

    # If boxes not found, fall back to first output
    if boxes is None and outputs:
        d, arr = outputs[0]
        boxes = arr[0] if arr.ndim == 3 else arr

    # Determine N
    N = boxes.shape[-2] if boxes is not None and boxes.ndim >= 2 else 0

    # Find tensors with N elements (scores and classes)
    candidate = []
    for d, arr in outputs:
        flat_n = arr.size
        if flat_n == N or (arr.ndim == 2 and arr.shape[-1] == N) or (arr.ndim == 2 and arr.shape[0] == N):
            candidate.append(arr)

    # If not enough candidates, try to include 1xN shapes
    if len(candidate) < 2:
        for d, arr in outputs:
            if arr.ndim == 2 and (arr.shape[0] == 1 and arr.shape[1] == N):
                candidate.append(arr)

    # Choose scores as the one within [0,1]
    chosen_scores = None
    chosen_classes = None
    for arr in candidate:
        a = arr
        if a.ndim == 2 and a.shape[0] == 1:
            a = a[0]
        if a.ndim > 1:
            continue
        if a.size == N:
            if np.all((a >= 0.0) & (a <= 1.0)):
                chosen_scores = a
            else:
                chosen_classes = a

    if chosen_scores is None or chosen_classes is None:
        # Fallback to common ordering: [boxes, classes, scores, num]
        if len(outputs) >= 3:
            boxes = outputs[0][1]
            classes = outputs[1][1]
            scores = outputs[2][1]
            if classes.ndim == 2 and classes.shape[0] == 1:
                classes = classes[0]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]
            if boxes.ndim == 3 and boxes.shape[0] == 1:
                boxes = boxes[0]
        else:
            # As last resort, return empty
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32), 0
    else:
        scores = chosen_scores
        classes = chosen_classes

    # num_detections if present
    for d, arr in outputs:
        if arr.size == 1:
            num = int(round(float(arr.flatten()[0])))
            break

    # Normalize shapes
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if classes.ndim == 2 and classes.shape[0] == 1:
        classes = classes[0]

    classes = classes.astype(np.int32)

    if num is None:
        num = N
    num = min(num, len(scores), len(classes), len(boxes))

    return boxes[:num], classes[:num], scores[:num], num

def draw_detections(frame, boxes, classes, scores, labels, conf_thresh, map_value):
    """
    Draws detection boxes and mAP text on frame.
    """
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf_thresh:
            continue
        cls_id = int(classes[i])
        label = labels.get(cls_id, str(cls_id))
        y_min, x_min, y_max, x_max = boxes[i]
        # Clip to [0,1]
        y_min = max(0.0, min(1.0, y_min))
        x_min = max(0.0, min(1.0, x_min))
        y_max = max(0.0, min(1.0, y_max))
        x_max = max(0.0, min(1.0, x_max))

        x1 = int(x_min * w)
        y1 = int(y_min * h)
        x2 = int(x_max * w)
        y2 = int(y_max * h)

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 1000.0)
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_bg_tl = (x1, max(0, y1 - th - 4))
        text_bg_br = (x1 + tw + 4, y1)
        cv2.rectangle(frame, text_bg_tl, text_bg_br, color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Draw mAP (proxy) on top-left corner
    map_text = f"mAP: {map_value:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w, h) / 800.0)
    thickness = 2
    cv2.putText(frame, map_text, (10, 30), font, font_scale, (50, 220, 255), thickness, cv2.LINE_AA)

def compute_proxy_map(confidence_history):
    """
    Computes a proxy mAP without ground truth, using detection confidences only.
    For each class, computes AP as the average precision across thresholds T in [0.5..0.95 step 0.05],
    where precision(T) = fraction of detections with confidence >= T.
    mAP is the mean over classes that had at least one detection.

    Note: This is NOT a true mAP metric (requires ground-truth). It is a proxy for demonstration.
    """
    thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50 to 0.95
    ap_list = []
    for cls_id, confs in confidence_history.items():
        if not confs:
            continue
        c = np.array(confs, dtype=np.float32)
        precisions = [(c >= t).mean() for t in thresholds]
        ap = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
        ap_list.append(ap)
    if not ap_list:
        return 0.0
    return float(np.mean(ap_list))

# =======================
# MAIN PIPELINE
# =======================
def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Setup interpreter (EdgeTPU)
    interpreter = make_interpreter(MODEL_PATH)
    input_index, in_h, in_w, in_dtype, in_quant = get_input_details(interpreter)

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open input video: {INPUT_PATH}")
        return

    # Prepare output video writer
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 0:
        input_fps = 30.0  # fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        # Read one frame to get size if properties unavailable
        ret_probe, probe_frame = cap.read()
        if not ret_probe:
            print("Error: failed to read from input video to determine frame size.")
            cap.release()
            return
        frame_h, frame_w = probe_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, input_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: cannot open video writer for: {OUTPUT_PATH}")
        cap.release()
        return

    # For proxy mAP computation: store confidences by class id
    confidence_history = {}

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess and set input
            set_input(interpreter, input_index, frame, in_h, in_w, in_dtype, in_quant)

            # Inference
            interpreter.invoke()

            # Get detections
            boxes, classes, scores, count = get_detections(interpreter)

            # Update confidence history for proxy mAP
            for i in range(count):
                cls_id = int(classes[i])
                score = float(scores[i])
                # collect all predicted confidences (even below display threshold) for proxy mAP
                if cls_id not in confidence_history:
                    confidence_history[cls_id] = []
                confidence_history[cls_id].append(score)

            # Compute proxy mAP so far
            proxy_map = compute_proxy_map(confidence_history)

            # Draw results
            draw_detections(frame, boxes, classes, scores, labels, CONF_THRESHOLD, proxy_map)

            # Write frame to output
            writer.write(frame)

            frame_count += 1
            if MAX_FRAMES > 0 and frame_count >= MAX_FRAMES:
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    final_map = compute_proxy_map(confidence_history)
    print(f"Processed {frame_count} frame(s) in {elapsed:.2f}s. Proxy mAP: {final_map:.3f}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()