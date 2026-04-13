import os
import sys
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports with fallback between tflite_runtime and tensorflow.lite
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_source = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        tflite_source = "tensorflow.lite"
    except Exception as e:
        print("ERROR: Unable to import TFLite Interpreter from either tflite_runtime or tensorflow.lite.")
        print(f"Details: {e}")
        sys.exit(1)

# Import cv2 only because the app explicitly needs video I/O and drawing
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is required for video I/O and drawing but is not available.")
    print(f"Details: {e}")
    sys.exit(1)

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
try:
    confidence_threshold = float('0.5')
except Exception:
    confidence_threshold = 0.5

# 1.3 Load Labels
def load_labels(labels_file):
    labels = []
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"WARNING: Could not load label file at {labels_file}. Using empty labels. Details: {e}")
        labels = []
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU delegate
def make_interpreter(model_file):
    delegate = None
    last_err = None
    # Attempt to load the EdgeTPU delegate from common names/paths
    for lib_name in ['libedgetpu.so.1.0', '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0']:
        try:
            delegate = load_delegate(lib_name)
            print(f"INFO: Loaded EdgeTPU delegate '{lib_name}' using {tflite_source}.")
            break
        except Exception as e:
            last_err = e
            delegate = None
    if delegate is None:
        print("WARNING: Failed to load EdgeTPU delegate. Inference may not work with an EdgeTPU-compiled model.")
        print("Please ensure the EdgeTPU runtime is installed. Tried: 'libedgetpu.so.1.0' and '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'.")
        print(f"Last delegate load error: {last_err}")

    try:
        if delegate is not None:
            interpreter = Interpreter(model_path=model_file, experimental_delegates=[delegate])
        else:
            interpreter = Interpreter(model_path=model_file)
        return interpreter
    except Exception as e:
        print("ERROR: Failed to create TFLite Interpreter. If this is an EdgeTPU-compiled model,")
        print("it typically requires the EdgeTPU delegate. Please install EdgeTPU runtime and retry.")
        print(f"Details: {e}")
        sys.exit(1)

interpreter = make_interpreter(model_path)
try:
    interpreter.allocate_tensors()
except Exception as e:
    print("ERROR: Failed to allocate tensors for the interpreter.")
    print("If using an EdgeTPU-compiled model without the EdgeTPU delegate, allocation will fail.")
    print(f"Details: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine input tensor properties
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)
in_height, in_width = int(input_shape[1]), int(input_shape[2])

# Utility: detection output parsing
def parse_detection_outputs(interpreter, output_details):
    """
    Attempts to find boxes, classes, scores, and count among the output tensors.
    Returns:
        boxes: np.ndarray shape (N, 4) with [ymin, xmin, ymax, xmax] normalized [0,1] if model outputs normalized.
        classes: np.ndarray shape (N,)
        scores: np.ndarray shape (N,)
        count: int number of valid detections (if provided) else derived from scores length
    """
    boxes = None
    classes = None
    scores = None
    count = None

    tensors = []
    for od in output_details:
        tensors.append(interpreter.get_tensor(od['index']))

    # Flatten batch dimension if present
    # Identify tensors based on shapes/dtypes heuristics
    for t in tensors:
        arr = t
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 4:
            boxes = arr[0]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            # Could be classes or scores
            flat = arr[0]
            if flat.dtype == np.float32 or flat.dtype == np.float16:
                scores = flat
            else:
                classes = flat
        elif arr.ndim == 1 and arr.shape[0] == 1:
            # num_detections
            try:
                count = int(arr[0])
            except Exception:
                pass

    # Some models may return boxes shape (N,4) directly without batch dim
    if boxes is None:
        for t in tensors:
            if t.ndim == 2 and t.shape[1] == 4:
                boxes = t

    # If classes are float, cast to int
    if classes is not None and np.issubdtype(classes.dtype, np.floating):
        classes = classes.astype(np.int32)

    # Fallbacks if count not provided
    if count is None:
        if scores is not None:
            count = scores.shape[0]
        elif boxes is not None:
            count = boxes.shape[0]
        else:
            count = 0

    # Ensure shapes are consistent up to count
    if boxes is not None and boxes.shape[0] < count:
        count = boxes.shape[0]
    if scores is not None and scores.shape[0] < count:
        count = scores.shape[0]
    if classes is not None and classes.shape[0] < count:
        count = classes.shape[0]

    # Slice to count
    if boxes is not None:
        boxes = boxes[:count]
    if classes is not None:
        classes = classes[:count]
    if scores is not None:
        scores = scores[:count]

    return boxes, classes, scores, count

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1 Acquire Input Data - open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"ERROR: Failed to open input video at {input_path}")
    sys.exit(1)

# Read first frame to initialize writer and check properties
ret, first_frame = cap.read()
if not ret or first_frame is None:
    print("ERROR: Could not read the first frame from the input video.")
    cap.release()
    sys.exit(1)

# Prepare VideoWriter with the same resolution and fps as input
frame_h, frame_w = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or np.isnan(fps):
    fps = 30.0  # fallback

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
if not out_writer.isOpened():
    print(f"ERROR: Failed to open output video for writing at {output_path}")
    cap.release()
    sys.exit(1)

# Stats for optional reporting
total_frames = 0
total_detections = 0
inference_times_ms = []

# Helper: Preprocess frame into model input tensor
def preprocess_frame_bgr(frame_bgr, target_w, target_h, floating):
    # Resize
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # Convert BGR -> RGB as most TFLite detection models expect RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)

    # 2.3 Quantization Handling
    if floating:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # For uint8 models, ensure dtype is uint8
        input_data = np.asarray(input_data, dtype=np.uint8)
    return input_data

# Main processing loop
current_frame = first_frame
while True:
    frame = current_frame

    # 2.2 Preprocess Data
    input_data = preprocess_frame_bgr(frame, in_width, in_height, floating_model)

    # Phase 3: Inference
    try:
        interpreter.set_tensor(input_index, input_data)
    except Exception as e:
        print("ERROR: Failed to set input tensor. Ensure input tensor shape and dtype match the model.")
        print(f"Details: {e}")
        break

    t0 = time.time()
    try:
        interpreter.invoke()
    except Exception as e:
        print("ERROR: Inference failed. If using an EdgeTPU-compiled model, the EdgeTPU delegate must be loaded.")
        print(f"Details: {e}")
        break
    t1 = time.time()
    inference_time_ms = (t1 - t0) * 1000.0
    inference_times_ms.append(inference_time_ms)

    # Phase 4: Output Interpretation & Handling
    # 4.1 Get Output Tensors
    boxes, classes, scores, count = parse_detection_outputs(interpreter, output_details)

    # 4.2 Interpret Results and 4.3 Post-processing (thresholding, scaling, clipping)
    detections = []
    if boxes is not None and scores is not None and classes is not None:
        for i in range(count):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            # Boxes assumed normalized [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[i]
            # Scale to image coordinates
            x1 = int(max(0, min(frame_w - 1, xmin * frame_w)))
            y1 = int(max(0, min(frame_h - 1, ymin * frame_h)))
            x2 = int(max(0, min(frame_w - 1, xmax * frame_w)))
            y2 = int(max(0, min(frame_h - 1, ymax * frame_h)))

            # Fix potential inverted coordinates
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            class_id = int(classes[i])
            label_name = labels[class_id] if (labels and 0 <= class_id < len(labels)) else f"id {class_id}"

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "score": score,
                "class_id": class_id,
                "label": label_name
            })

    # 4.4 Handle Output - draw rectangles and labels; overlay mAP (N/A without GT)
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        score = det["score"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        caption = f"{label}: {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), (0, 255, 0), thickness=-1)
        cv2.putText(frame, caption, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    total_detections += len(detections)
    total_frames += 1

    # Overlay inference time and "mAP" info (no GT available -> N/A)
    info_line1 = f"Inference: {inference_time_ms:.1f} ms   Dets: {len(detections)}"
    info_line2 = "mAP: N/A (no ground truth annotations provided)"
    cv2.putText(frame, info_line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, info_line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)

    # Write frame to output
    out_writer.write(frame)

    # 4.5 Loop Continuation
    ret, next_frame = cap.read()
    if not ret or next_frame is None:
        break
    current_frame = next_frame

# Phase 5: Cleanup
cap.release()
out_writer.release()

# Final reporting
if total_frames > 0:
    avg_inf = np.mean(inference_times_ms) if inference_times_ms else 0.0
    print(f"Processing complete.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {total_frames}")
    print(f"Total detections (score >= {confidence_threshold}): {total_detections}")
    print(f"Average inference time per frame: {avg_inf:.2f} ms")
    print("mAP: N/A (no ground truth available to compute mean Average Precision).")
else:
    print("No frames processed. Please check the input video file.")