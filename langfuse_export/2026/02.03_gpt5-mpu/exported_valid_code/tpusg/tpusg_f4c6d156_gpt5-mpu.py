import os
import time
import numpy as np
import cv2

# Phase 1: Setup

# 1.1 Imports
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        TFLITE_SOURCE = "tensorflow.lite"
    except Exception as e:
        raise ImportError("Failed to import TFLite Interpreter from both tflite_runtime and tensorflow.lite.") from e

# 1.2 Paths/Parameters (use provided configuration)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold_str  = 0.5
try:
    confidence_threshold = float(confidence_threshold_str)
except ValueError as e:
    raise ValueError(f"Invalid confidence_threshold value: {confidence_threshold_str}") from e

# 1.3 Load Labels
def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line)
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
interpreter = None
delegates_info = []
delegate_errors = []

def try_make_interpreter_with_delegate(delegate_lib):
    try:
        intr = Interpreter(model_path=model_path, experimental_delegates=[load_delegate(delegate_lib)])
        return intr, None
    except Exception as e:
        return None, e

# Try standard EdgeTPU shared object names typically available on Coral Dev Board
candidate_delegates = [
    'libedgetpu.so.1.0',
    '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'
]

for lib in candidate_delegates:
    intr, err = try_make_interpreter_with_delegate(lib)
    if intr is not None:
        interpreter = intr
        delegates_info.append(f"Using EdgeTPU delegate: {lib}")
        break
    else:
        delegate_errors.append((lib, err))

if interpreter is None:
    # EdgeTPU delegate failed: provide informative errors and fall back to CPU
    error_msg_lines = ["Failed to load EdgeTPU delegate. Detailed attempts:"]
    for lib, err in delegate_errors:
        error_msg_lines.append(f" - {lib}: {repr(err)}")
    error_msg_lines.append("Falling back to CPU interpreter (performance will be reduced).")
    print("\n".join(error_msg_lines))
    interpreter = Interpreter(model_path=model_path)

# Allocate tensors
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Utility: Determine model input requirements
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]
input_height, input_width = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Utility: Prepare output parsing helper
def parse_detection_outputs(interpreter, output_details):
    """
    Retrieve detection outputs and return boxes, classes, scores, num_detections.
    - boxes: (N, 4) with [ymin, xmin, ymax, xmax] normalized [0,1]
    - classes: (N,) int
    - scores: (N,) float
    - num_detections: int
    """
    raw_outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    boxes = None
    classes = None
    scores = None
    num = None

    # Identify outputs by shape heuristics
    for out in raw_outputs:
        out_arr = np.array(out)
        if out_arr.ndim == 3 and out_arr.shape[0] == 1 and out_arr.shape[2] == 4 and out_arr.dtype == np.float32:
            boxes = out_arr[0]
        elif out_arr.ndim == 2 and out_arr.shape[0] == 1 and out_arr.dtype in (np.float32, np.int32, np.int64):
            # Could be classes or scores
            vec = out_arr[0]
            # Scores typically [0, 1]
            if np.all(vec >= 0.0) and np.all(vec <= 1.0):
                scores = vec.astype(np.float32)
            else:
                classes = vec.astype(np.int32)
        elif out_arr.ndim == 2 and out_arr.shape == (1, 1):
            # Some models use (1,1) for num_detections
            num = int(out_arr.flatten()[0])
        elif out_arr.ndim == 1 and out_arr.shape[0] == 1:
            num = int(out_arr[0])

    # Fallback: if num not set, infer from boxes/scores length
    if num is None:
        if boxes is not None:
            num = boxes.shape[0]
        elif scores is not None:
            num = scores.shape[0]
        elif classes is not None:
            num = classes.shape[0]
        else:
            num = 0

    # Basic sanity defaults
    if boxes is None:
        boxes = np.zeros((num, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((num,), dtype=np.int32)
    if scores is None:
        scores = np.zeros((num,), dtype=np.float32)

    # Ensure consistent lengths
    n = min(num, boxes.shape[0], classes.shape[0], scores.shape[0])
    return boxes[:n], classes[:n], scores[:n], n

# Video utility: create writer with source properties
def create_video_writer(out_path, frame_width, frame_height, fps):
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    return writer

# Drawing utility
def draw_detections(frame_bgr, detections, label_list, conf_thresh):
    """
    Draw bounding boxes and labels on frame_bgr.
    detections: list of dict with keys ['ymin','xmin','ymax','xmax','score','class_id','label']
    """
    for det in detections:
        score = det['score']
        if score < conf_thresh:
            continue
        ymin, xmin, ymax, xmax = det['ymin'], det['xmin'], det['ymax'], det['xmax']
        class_id = det['class_id']
        label = det['label']

        # Draw rectangle
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Prepare label text
        text = f"{label}: {score:.2f}"
        # Put a filled rectangle for text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (xmin, ymin - th - baseline - 4), (xmin + tw + 2, ymin), (0, 255, 0), thickness=-1)
        cv2.putText(frame_bgr, text, (xmin + 1, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Failed to open input video: {input_path}")

src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or src_fps <= 0.0 or np.isnan(src_fps):
    src_fps = 30.0  # default fallback

src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = create_video_writer(output_path, src_width, src_height, src_fps)
if not writer.isOpened():
    cap.release()
    raise IOError(f"Failed to create output video writer: {output_path}")

# Informative logs
if delegates_info:
    print(delegates_info[0])
else:
    print("Running inference without EdgeTPU delegate (CPU fallback).")
print(f"TFLite source: {TFLITE_SOURCE}")
print(f"Model: {model_path}")
print(f"Labels: {label_path}")
print(f"Input video: {input_path}")
print(f"Output video: {output_path}")
print(f"Confidence threshold: {confidence_threshold}")

frame_index = 0
inference_times = []

# Since no ground truth annotations are provided, mAP cannot be computed meaningfully.
# We will annotate frames with an explicit note about mAP unavailability.
map_text_overlay = "mAP: N/A (no ground truth provided)"

start_time_total = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break  # End of video

    original_frame = frame_bgr.copy()

    # 2.2 Preprocess Data
    # Convert BGR to RGB as most TFLite models expect RGB input
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)

    # 2.3 Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Ensure dtype matches model input; many EdgeTPU models expect uint8
        input_data = np.asarray(input_data, dtype=input_dtype)

    # Phase 3: Inference
    # 3.1 Set Input Tensor
    interpreter.set_tensor(input_index, input_data)

    # 3.2 Run Inference
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    inference_times.append((t1 - t0) * 1000.0)  # ms

    # Phase 4: Output Interpretation & Handling
    # 4.1 Get Output Tensors
    boxes_norm, classes_ids, scores_vals, num_dets = parse_detection_outputs(interpreter, output_details)

    # 4.2 Interpret Results: map class IDs to labels
    detections = []
    for i in range(num_dets):
        score = float(scores_vals[i])
        class_id = int(classes_ids[i])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"ID {class_id}"

        # 4.3 Post-processing: confidence thresholding and box scaling + clipping
        y_min = float(boxes_norm[i][0])
        x_min = float(boxes_norm[i][1])
        y_max = float(boxes_norm[i][2])
        x_max = float(boxes_norm[i][3])

        # Scale to original frame size
        xmin_px = max(0, min(src_width - 1, int(x_min * src_width)))
        ymin_px = max(0, min(src_height - 1, int(y_min * src_height)))
        xmax_px = max(0, min(src_width - 1, int(x_max * src_width)))
        ymax_px = max(0, min(src_height - 1, int(y_max * src_height)))

        # Ensure proper box ordering
        xmin_px, xmax_px = (xmin_px, xmax_px) if xmin_px <= xmax_px else (xmax_px, xmin_px)
        ymin_px, ymax_px = (ymin_px, ymax_px) if ymin_px <= ymax_px else (ymax_px, ymin_px)

        detections.append({
            'xmin': xmin_px,
            'ymin': ymin_px,
            'xmax': xmax_px,
            'ymax': ymax_px,
            'score': score,
            'class_id': class_id,
            'label': label
        })

    # 4.4 Handle Output: draw detections and write frame to video
    draw_detections(frame_bgr, detections, labels, confidence_threshold)

    # Overlay mAP information (explicitly marked as unavailable due to lack of ground truth)
    cv2.putText(frame_bgr, map_text_overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 255), 2, cv2.LINE_AA)

    # Optionally, overlay simple runtime info
    if inference_times:
        ms = inference_times[-1]
        cv2.putText(frame_bgr, f"Inference: {ms:.1f} ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)

    writer.write(frame_bgr)

    frame_index += 1

# Phase 5: Cleanup
cap.release()
writer.release()
total_time = time.time() - start_time_total

# Summary logs
if inference_times:
    avg_ms = float(np.mean(inference_times))
    print(f"Processed {frame_index} frames in {total_time:.2f} s ({frame_index/total_time:.2f} FPS).")
    print(f"Average inference time: {avg_ms:.2f} ms")
else:
    print("No frames were processed.")
print("Note: mAP calculation was skipped because no ground truth annotations were provided for the input video.")