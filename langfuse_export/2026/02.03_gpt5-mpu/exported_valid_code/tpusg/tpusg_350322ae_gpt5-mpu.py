import os
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports (with fallback)
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    interpreter_source = 'tflite_runtime'
except Exception as e_rt:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
        interpreter_source = 'tensorflow.lite'
    except Exception as e_tf:
        raise SystemExit(
            f"ERROR: Failed to import TFLite Interpreter from both tflite_runtime and tensorflow.lite.\n"
            f"tflite_runtime error: {e_rt}\n"
            f"tensorflow.lite error: {e_tf}"
        )

# Import only if image/video processing is explicitly mentioned
import cv2

# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3 Load Labels (Conditional)
labels = []
if os.path.isfile(label_path):
    try:
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"WARNING: Failed to read label file at '{label_path}': {e}")
else:
    print(f"WARNING: Label file not found at '{label_path}'. Class IDs will be shown without names.")

# 1.4 Load Interpreter with EdgeTPU
interpreter = None
delegate_errors = []
delegate_candidates = ['libedgetpu.so.1.0', '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0']
for lib in delegate_candidates:
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(lib)]
        )
        break
    except Exception as e:
        delegate_errors.append(f"{lib}: {e}")

if interpreter is None:
    raise SystemExit(
        "ERROR: Failed to load the EdgeTPU delegate. Attempted libraries:\n  - "
        + "\n  - ".join(delegate_candidates)
        + "\nDetails:\n  - " + "\n  - ".join(delegate_errors)
        + "\nEnsure that the Coral EdgeTPU runtime is installed and accessible."
    )

try:
    interpreter.allocate_tensors()
except Exception as e:
    raise SystemExit(f"ERROR: Failed to allocate tensors for the interpreter: {e}")

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    raise SystemExit("ERROR: Model has no input tensors.")

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_height = int(input_shape[1])
input_width = int(input_shape[2])
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Utility: Map model outputs to boxes/scores/classes/count robustly
def parse_detection_outputs(output_details_list, interpreter_obj):
    outputs = [interpreter_obj.get_tensor(od['index']) for od in output_details_list]
    boxes = None
    scores = None
    classes = None
    count = None

    # Identify outputs by shape/dtype heuristics
    for arr in outputs:
        arr_np = np.squeeze(arr)
        if arr_np.ndim == 2 and arr_np.shape[-1] == 4:
            boxes = arr
        elif arr_np.ndim == 1 and arr_np.size == 1:
            count = arr
        elif arr_np.ndim == 2:
            # Could be scores or classes
            if arr.dtype == np.float32 or arr.dtype == np.float16:
                if scores is None:
                    scores = arr
            else:
                if classes is None:
                    classes = arr

    # Some models might return 1xN for classes as float (rare)
    if scores is None or classes is None or boxes is None:
        # Try alternative mapping by names if available
        for od in output_details_list:
            name = od.get('name', '').lower()
            tensor = interpreter_obj.get_tensor(od['index'])
            if 'box' in name and tensor.shape[-1] == 4:
                boxes = tensor
            elif 'class' in name:
                classes = tensor
            elif 'score' in name:
                scores = tensor
            elif 'count' in name or 'num' in name:
                count = tensor

    # Final sanity fallbacks
    if boxes is None:
        raise RuntimeError("Failed to identify 'boxes' output tensor.")
    if scores is None:
        raise RuntimeError("Failed to identify 'scores' output tensor.")
    if classes is None:
        raise RuntimeError("Failed to identify 'classes' output tensor.")

    # If count is missing, infer from scores length
    if count is None:
        count_val = scores.shape[1] if scores.ndim == 2 else scores.shape[0]
        count = np.array([count_val], dtype=np.int32).reshape((1,))

    return boxes, scores, classes, count

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1 Acquire Input Data (single video file)
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit(f"ERROR: Failed to open input video: {input_path}")

# Prepare output writer
input_fps = cap.get(cv2.CAP_PROP_FPS)
if not input_fps or np.isnan(input_fps) or input_fps <= 0:
    input_fps = 30.0  # fallback if FPS is unavailable
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
out_writer = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
if not out_writer.isOpened():
    cap.release()
    raise SystemExit(f"ERROR: Failed to open output video for writing: {output_path}")

# Prepare timers/statistics
total_frames = 0
inference_times = []

# Helper for preprocessing
def preprocess_frame_bgr_to_input(frame_bgr, target_w, target_h, floating):
    # Resize to model input
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Expand to NHWC
    input_tensor = np.expand_dims(rgb, axis=0)
    if floating:
        input_tensor = input_tensor.astype(np.float32)
        input_tensor = (input_tensor - 127.5) / 127.5
    else:
        # Ensure uint8 for quantized models
        input_tensor = input_tensor.astype(np.uint8)
    return input_tensor

# Drawing utilities
def draw_detections_on_frame(frame_bgr, detections, labels_list):
    for det in detections:
        x_min, y_min, x_max, y_max, score, class_id = det
        # Draw bounding box
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Prepare label text
        class_id_int = int(class_id)
        if 0 <= class_id_int < len(labels_list):
            class_name = labels_list[class_id_int]
        else:
            class_name = f"id:{class_id_int}"

        label_text = f"{class_name} {int(score * 100)}%"
        # Text background for readability
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x_min, y_min - text_h - baseline), (x_min + text_w, y_min), (0, 255, 0), -1)
        cv2.putText(frame_bgr, label_text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 2.4 Loop Control: process all frames until video ends
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    total_frames += 1

    # 2.2 Preprocess Data
    input_data = preprocess_frame_bgr_to_input(frame_bgr, input_width, input_height, floating_model)

    # 2.3 Quantization Handling already done in preprocess

    # Phase 3: Inference
    # 3.1 Set Input Tensor
    interpreter.set_tensor(input_index, input_data)
    # 3.2 Run Inference
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    inference_times.append((t1 - t0) * 1000.0)  # milliseconds

    # Phase 4: Output Interpretation & Handling Loop
    # 4.1 Get Output Tensor(s)
    try:
        boxes_raw, scores_raw, classes_raw, count_raw = parse_detection_outputs(output_details, interpreter)
    except Exception as e:
        # In case of any unexpected output parsing issue, continue safely
        print(f"WARNING: Failed to parse detection outputs on frame {total_frames}: {e}")
        out_writer.write(frame_bgr)
        continue

    # 4.2 Interpret Results
    # Squeeze to remove batch dimension
    boxes = np.squeeze(boxes_raw)
    scores = np.squeeze(scores_raw)
    classes = np.squeeze(classes_raw)
    if count_raw is not None:
        det_count = int(np.squeeze(count_raw))
    else:
        det_count = scores.shape[0] if scores.ndim == 1 else scores.shape[1]

    # 4.3 Post-processing: confidence thresholding, scaling, clipping
    H, W = frame_bgr.shape[:2]
    detections_to_draw = []
    for i in range(det_count):
        if scores.ndim == 1:
            score = float(scores[i])
        else:
            score = float(scores[0][i])
        if score < confidence_threshold:
            continue

        if boxes.ndim == 2:
            y_min, x_min, y_max, x_max = boxes[i]
        else:
            y_min, x_min, y_max, x_max = boxes[0][i]

        # Coordinates may be normalized [0,1], scale to frame size
        x_min_abs = int(max(0, min(W - 1, x_min * W)))
        x_max_abs = int(max(0, min(W - 1, x_max * W)))
        y_min_abs = int(max(0, min(H - 1, y_min * H)))
        y_max_abs = int(max(0, min(H - 1, y_max * H)))

        # Ensure proper box ordering
        x1, x2 = sorted([x_min_abs, x_max_abs])
        y1, y2 = sorted([y_min_abs, y_max_abs])

        if classes.ndim == 1:
            class_id = int(classes[i])
        else:
            class_id = int(classes[0][i])

        detections_to_draw.append((x1, y1, x2, y2, score, class_id))

    # 4.4 Handle Output: draw detections and mAP text, then write frame
    draw_detections_on_frame(frame_bgr, detections_to_draw, labels)

    # mAP calculation requires ground truth annotations, which are not provided.
    # To satisfy the output requirement, we display that mAP is not available.
    map_text = "mAP: N/A (no ground truth)"
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)

    # Write annotated frame to output
    out_writer.write(frame_bgr)

# Phase 5: Cleanup
cap.release()
out_writer.release()

# Print a brief summary
if inference_times:
    avg_inf_ms = sum(inference_times) / len(inference_times)
    print(f"Completed processing {total_frames} frames.")
    print(f"Average inference time: {avg_inf_ms:.2f} ms per frame (on EdgeTPU).")
else:
    print("No frames were processed from the input video.")

print(f"Annotated output video saved to: {output_path}")