import os
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports: Interpreter and delegate with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    RUNTIME_SOURCE = 'tflite_runtime'
except ImportError:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
        RUNTIME_SOURCE = 'tensorflow'
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to import TFLite Interpreter. Details: {e}")

# 1.2 Paths/Parameters (use provided configuration parameters exactly)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1.3 Load Labels (if needed)
labels = []
if label_path and os.path.isfile(label_path):
    try:
        with open(label_path, 'r') as lf:
            for line in lf:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"WARNING: Unable to load label file at {label_path}. Details: {e}")
        labels = []
else:
    print(f"WARNING: Label file not found at {label_path}. Proceeding without class names.")
    labels = []

# 1.4 Load Interpreter with EdgeTPU delegate and error handling
interpreter = None
delegate_status = 'cpu'
delegate_error_msgs = []

try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
    delegate_status = 'edgetpu'
except Exception as e1:
    delegate_error_msgs.append(str(e1))
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
        delegate_status = 'edgetpu'
    except Exception as e2:
        delegate_error_msgs.append(str(e2))
        print("WARNING: Failed to load EdgeTPU delegate. Falling back to CPU.")
        print("Details:")
        for i, msg in enumerate(delegate_error_msgs, 1):
            print(f"  Attempt {i} error: {msg}")
        print("If you intend to use EdgeTPU, please ensure libedgetpu is installed and the model is EdgeTPU-compiled.")
        try:
            interpreter = Interpreter(model_path=model_path)
        except Exception as e3:
            raise SystemExit(f"ERROR: Failed to create TFLite Interpreter. Details: {e3}")

# Allocate tensors
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_height, input_width = int(input_shape[1]), int(input_shape[2])
floating_model = (input_dtype == np.float32)

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire Input Data: read a single video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit(f"ERROR: Failed to open input video: {input_path}")

# Prepare VideoWriter for output
input_fps = cap.get(cv2.CAP_PROP_FPS)
if not input_fps or input_fps <= 0:
    input_fps = 30.0  # Default fallback
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
if not out_writer.isOpened():
    raise SystemExit(f"ERROR: Failed to open output video for writing: {output_path}")

# Tracking for proxy mAP calculation (no ground truth provided)
all_detection_confidences = []
total_frames = 0
inference_times_ms = []

def preprocess_frame(frame_bgr):
    """
    Resize and convert frame to model's expected input format.
    Returns input_data ready to feed into TFLite interpreter.
    """
    # 2.2 Preprocess Data: Resize and color convert BGR -> RGB
    resized = cv2.resize(frame_bgr, (input_width, input_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Add batch dimension
    input_data = np.expand_dims(rgb, axis=0)

    # 2.3 Quantization Handling
    if floating_model:
        # Normalize to [-1, 1]
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Ensure dtype matches (typically uint8 for EdgeTPU models)
        input_data = np.asarray(input_data, dtype=input_dtype)

    return input_data

# =========================
# Phase 3: Inference
# =========================

def run_inference(input_data):
    """
    Set input tensor and invoke inference.
    """
    # 3.1 Set Input Tensor
    interpreter.set_tensor(input_index, input_data)

    # 3.2 Run Inference
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    inference_times_ms.append((t1 - t0) * 1000.0)

# =========================
# Phase 4: Output Interpretation & Handling Loop
# =========================

def get_output_tensors():
    """
    4.1 Get Output Tensor(s) and return as (boxes, classes, scores, num_detections).
    Attempts standard SSD order; falls back to heuristic if necessary.
    """
    # Try standard order: [boxes, classes, scores, num]
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = interpreter.get_tensor(output_details[3]['index'])
        # Remove batch dimension if present
        if boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes[0]
        if classes.ndim == 2 and classes.shape[0] == 1:
            classes = classes[0]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        if isinstance(num, np.ndarray):
            num = int(np.squeeze(num))
        else:
            num = int(num)
        return boxes, classes, scores, num
    except Exception:
        # Heuristic fallback
        outputs = [interpreter.get_tensor(od['index']) for od in output_details]
        processed = []
        for out in outputs:
            arr = out
            if arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr[0]
            processed.append(arr)

        boxes = None
        classes = None
        scores = None
        num = None

        for arr in processed:
            if isinstance(arr, np.ndarray):
                if arr.ndim == 2 and arr.shape[1] == 4:
                    boxes = arr
        for arr in processed:
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                if arr.dtype in (np.float32, np.float64) and np.all((arr >= 0.0) & (arr <= 1.0)):
                    # Candidate scores
                    if scores is None or len(arr) >= len(scores):
                        scores = arr
        for arr in processed:
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                if arr.dtype in (np.float32, np.float64, np.int32, np.int64):
                    if arr is not scores:
                        if classes is None or len(arr) >= len(classes):
                            classes = arr
        for arr in processed:
            if isinstance(arr, np.ndarray) and arr.size == 1:
                value = int(np.squeeze(arr))
                if 0 <= value <= 1000:
                    num = value

        if boxes is None or classes is None or scores is None:
            raise RuntimeError("ERROR: Unable to parse model outputs for detection.")

        if num is None:
            num = min(len(scores), len(classes), boxes.shape[0])

        return boxes, classes, scores, int(num)

def overlay_detections_and_metrics(frame_bgr, detections, map_proxy_value):
    """
    Draw bounding boxes and labels on the frame. Also overlay proxy mAP value.
    detections: list of dicts with keys: (ymin, xmin, ymax, xmax, class_id, score, label)
    """
    # Draw detections
    for det in detections:
        ymin, xmin, ymax, xmax = det['ymin'], det['xmin'], det['ymax'], det['xmax']
        class_id = det['class_id']
        score = det['score']
        label_text = det['label']

        # Rectangle
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Label
        text = f"{label_text}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(ymin - 10, text_h + 5)
        cv2.rectangle(frame_bgr, (xmin, y_text - text_h - baseline), (xmin + text_w, y_text + baseline // 2), (0, 255, 0), -1)
        cv2.putText(frame_bgr, text, (xmin, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay proxy mAP at top-left
    map_text = f"mAP (proxy): {map_proxy_value:.3f}"
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2, cv2.LINE_AA)

    # Overlay delegate info and FPS
    avg_inf_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    fps_text = f"Inference: {avg_inf_ms:.1f} ms | Delegate: {delegate_status}"
    cv2.putText(frame_bgr, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 2, cv2.LINE_AA)

    return frame_bgr

def interpret_and_postprocess(outputs, frame_shape):
    """
    4.2 Interpret Results and 4.3 Post-processing:
    - Process raw outputs into detection results
    - Apply confidence thresholding
    - Scale bounding boxes to frame coordinates and clip
    """
    boxes, classes, scores, num = outputs
    fh, fw = frame_shape[0], frame_shape[1]

    dets = []
    count = min(num, boxes.shape[0], len(scores), len(classes))
    for i in range(count):
        score = float(scores[i])
        if score < confidence_threshold:
            continue

        # Box order: [ymin, xmin, ymax, xmax] in normalized coordinates
        box = boxes[i]
        ymin = int(max(0.0, min(1.0, float(box[0]))) * fh)
        xmin = int(max(0.0, min(1.0, float(box[1]))) * fw)
        ymax = int(max(0.0, min(1.0, float(box[2]))) * fh)
        xmax = int(max(0.0, min(1.0, float(box[3]))) * fw)

        # Clip
        ymin = max(0, min(fh - 1, ymin))
        xmin = max(0, min(fw - 1, xmin))
        ymax = max(0, min(fh - 1, ymax))
        xmax = max(0, min(fw - 1, xmax))

        # Ensure proper box
        if xmax <= xmin or ymax <= ymin:
            continue

        class_id = int(classes[i]) if classes is not None else -1
        if 0 <= class_id < len(labels):
            label_text = labels[class_id]
        else:
            label_text = f"id_{class_id}"

        dets.append({
            'ymin': ymin, 'xmin': xmin, 'ymax': ymax, 'xmax': xmax,
            'class_id': class_id, 'score': score, 'label': label_text
        })

    return dets

# Processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    total_frames += 1

    # Preprocess
    input_data = preprocess_frame(frame)

    # Inference
    run_inference(input_data)

    # 4.1-4.3: Get outputs, interpret, and postprocess
    outputs = get_output_tensors()
    detections = interpret_and_postprocess(outputs, frame.shape)

    # Aggregate confidences for proxy mAP calculation
    for det in detections:
        all_detection_confidences.append(det['score'])

    # Compute proxy mAP as mean of detection confidences (no ground truth available)
    map_proxy = float(np.mean(all_detection_confidences)) if all_detection_confidences else 0.0

    # 4.4 Handle Output: draw detections and metrics, write frame
    annotated_frame = overlay_detections_and_metrics(frame.copy(), detections, map_proxy)
    out_writer.write(annotated_frame)

# 4.5 Loop Continuation handled by video end condition

# =========================
# Phase 5: Cleanup
# =========================

cap.release()
out_writer.release()

final_map_proxy = float(np.mean(all_detection_confidences)) if all_detection_confidences else 0.0
avg_infer_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
print("Processing complete.")
print(f"Input video: {input_path}")
print(f"Output video with detections: {output_path}")
print(f"Frames processed: {total_frames}")
print(f"Average inference time: {avg_infer_ms:.2f} ms | Delegate used: {delegate_status}")
print(f"Final mAP (proxy, no ground truth provided): {final_map_proxy:.4f}")