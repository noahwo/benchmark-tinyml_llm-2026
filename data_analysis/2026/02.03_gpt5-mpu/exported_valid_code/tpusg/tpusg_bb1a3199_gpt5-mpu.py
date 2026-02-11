import os
import time
import numpy as np

# Phase 1: Setup

# 1.1 Imports: Try tflite_runtime first, then fallback to tensorflow.lite
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_src = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        tflite_src = "tensorflow.lite"
    except Exception as e:
        print("ERROR: Neither tflite_runtime nor tensorflow.lite could be imported.")
        print("Detail:", repr(e))
        raise SystemExit(1)

# Import cv2 for video processing (required by application)
import cv2

# 1.2 Paths/Parameters (Use provided configuration exactly)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5
CONF_THRESH = float(confidence_threshold)

# 1.3 Load Labels (if provided and relevant)
def load_labels_file(label_file_path):
    labels_list = []
    if not os.path.exists(label_file_path):
        print(f"WARNING: Label file not found at: {label_file_path}. Proceeding with empty labels.")
        return labels_list
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label = line.strip()
                if label != "":
                    labels_list.append(label)
    except Exception as e:
        print(f"WARNING: Failed to read label file at {label_file_path}: {repr(e)}")
    return labels_list

labels = load_labels_file(label_path)

# 1.4 Load Interpreter with EdgeTPU delegate and error handling
interpreter = None
delegate_errors = []
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e1:
    delegate_errors.append(("libedgetpu.so.1.0", repr(e1)))
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e2:
        delegate_errors.append(("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0", repr(e2)))
        print("ERROR: Failed to load EdgeTPU delegate. This application requires EdgeTPU acceleration.")
        print("Attempted delegates and errors:")
        for lib, err in delegate_errors:
            print(f" - {lib}: {err}")
        print("Ensure the EdgeTPU runtime is installed and the correct delegate library is available.")
        raise SystemExit(1)

# Allocate tensors
try:
    interpreter.allocate_tensors()
except Exception as e:
    print("ERROR: Failed to allocate tensors for the TFLite interpreter.")
    print("Detail:", repr(e))
    raise SystemExit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Validate single input tensor
if len(input_details) != 1:
    print(f"ERROR: Expected exactly 1 input tensor, found {len(input_details)}.")
    raise SystemExit(1)

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # Expect [1, H, W, 3]
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Validate input tensor shape and get model input size as integers
if not (len(input_shape) == 4 and input_shape[0] == 1 and input_shape[3] == 3):
    print(f"ERROR: Unsupported input tensor shape: {input_shape}. Expected [1, H, W, 3].")
    raise SystemExit(1)

try:
    model_in_height = int(input_shape[1])
    model_in_width = int(input_shape[2])
except Exception as e:
    print("ERROR: Failed to parse model input dimensions as integers.")
    print("Detail:", repr(e))
    raise SystemExit(1)

# Utility: Map class id to label robustly (handles 0-based and 1-based indices)
def get_label_for_class_id(class_id, labels_list):
    default_label = f"id_{class_id}"
    if not labels_list:
        return default_label
    # Try 0-based first
    if 0 <= class_id < len(labels_list):
        return labels_list[class_id]
    # Try 1-based indexing (common in some detection models)
    if 1 <= class_id <= len(labels_list):
        return labels_list[class_id - 1]
    return default_label

# Phase 2: Input Acquisition & Preprocessing Loop

# 2.1 Acquire Input Data: Open the input video file
if not os.path.exists(input_path):
    print(f"ERROR: Input video file not found: {input_path}")
    raise SystemExit(1)

video_cap = cv2.VideoCapture(input_path)
if not video_cap.isOpened():
    print(f"ERROR: Failed to open video file: {input_path}")
    raise SystemExit(1)

# Retrieve original video properties for output writer
orig_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_val = video_cap.get(cv2.CAP_PROP_FPS)
try:
    fps = float(fps_val)
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0  # Fallback FPS if metadata is invalid
except Exception:
    fps = 30.0

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
if not video_writer.isOpened():
    print(f"ERROR: Failed to open output video file for writing: {output_path}")
    video_cap.release()
    raise SystemExit(1)

# Statistics for proxy mAP (mean of per-class average confidences above threshold)
per_class_scores = {}  # label -> list of confidences
total_frames = 0
total_infer_time = 0.0

# Helper to parse typical TFLite Detection PostProcess outputs
def parse_detection_outputs(interpreter_obj, output_info):
    # Typical outputs: boxes: [1, N, 4], classes: [1, N], scores: [1, N], count: [1]
    outs = [interpreter_obj.get_tensor(od['index']) for od in output_info]

    # Squeeze each output for easier handling
    flat = [np.squeeze(o) for o in outs]

    boxes = None
    classes = None
    scores = None
    count = None

    # Identify boxes tensor (2D with last dim == 4) or (N,4)
    for arr in flat:
        if arr.ndim == 2 and arr.shape[1] == 4:
            boxes = arr.astype(np.float32)
            break

    # Identify scores and classes (1D arrays length N)
    for arr in flat:
        if boxes is not None and arr is boxes:
            continue
        if arr.ndim == 1 and arr.size > 0:
            # scores usually in [0,1]
            minv = float(np.min(arr))
            maxv = float(np.max(arr))
            if 0.0 <= minv and maxv <= 1.0 and scores is None:
                scores = arr.astype(np.float32)
            elif classes is None:
                classes = arr.astype(np.float32)  # may be float, will cast to int later

    # Identify count scalar
    for arr in flat:
        if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
            try:
                count = int(arr if arr.ndim == 0 else arr[0])
                break
            except Exception:
                continue

    # Fallbacks
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.float32)
    if count is None:
        count = min(len(scores), len(classes), boxes.shape[0])

    # Ensure shapes are consistent
    n = min(count, boxes.shape[0], scores.shape[0], classes.shape[0])
    boxes = boxes[:n]
    scores = scores[:n]
    classes = classes[:n]
    count = n

    return boxes, classes, scores, count

# Main processing loop (read the single video file frame-by-frame)
while True:
    ret, frame_bgr = video_cap.read()
    if not ret:
        break  # End of video stream

    total_frames += 1
    orig_h, orig_w = frame_bgr.shape[:2]

    # Phase 2.2: Preprocess Data (BGR->RGB, resize to model input, add batch dimension)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(frame_rgb, (model_in_width, model_in_height), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized_rgb, axis=0)

    # Phase 2.3: Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = input_data.astype(input_dtype)

    # Phase 3: Inference
    interpreter.set_tensor(input_index, input_data)
    t0 = time.time()
    interpreter.invoke()
    infer_time = time.time() - t0
    total_infer_time += infer_time

    # Phase 4: Output Interpretation & Handling

    # 4.1 Get Output Tensors
    boxes, classes, scores, count = parse_detection_outputs(interpreter, output_details)

    # 4.2 Interpret Results & 4.3 Post-processing: thresholding, scaling, clipping
    drawn = 0
    for i in range(count):
        score = float(scores[i])
        if score < CONF_THRESH:
            continue

        # Boxes are typically [ymin, xmin, ymax, xmax] normalized to [0,1]
        y_min, x_min, y_max, x_max = [float(v) for v in boxes[i]]

        # Clip to [0,1]
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        # Scale to pixel coordinates
        x_min_abs = int(round(x_min * orig_w))
        y_min_abs = int(round(y_min * orig_h))
        x_max_abs = int(round(x_max * orig_w))
        y_max_abs = int(round(y_max * orig_h))

        # Ensure bbox is within image bounds
        x_min_abs = max(0, min(orig_w - 1, x_min_abs))
        y_min_abs = max(0, min(orig_h - 1, y_min_abs))
        x_max_abs = max(0, min(orig_w - 1, x_max_abs))
        y_max_abs = max(0, min(orig_h - 1, y_max_abs))

        # Validate bbox size
        if x_max_abs <= x_min_abs or y_max_abs <= y_min_abs:
            continue

        class_id = int(classes[i])
        label_text = get_label_for_class_id(class_id, labels)

        # Draw bounding box
        cv2.rectangle(frame_bgr, (x_min_abs, y_min_abs), (x_max_abs, y_max_abs), (0, 255, 0), 2)

        # Draw label with confidence
        caption = f"{label_text}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x_min_abs
        text_y = max(0, y_min_abs - 5)
        cv2.rectangle(frame_bgr, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), (0, 0, 0), thickness=-1)
        cv2.putText(frame_bgr, caption, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Update proxy mAP statistics (mean of per-class average confidences)
        if label_text not in per_class_scores:
            per_class_scores[label_text] = []
        per_class_scores[label_text].append(score)

        drawn += 1

    # Compute proxy mAP (mean of per-class average confidences)
    per_class_ap = []
    for cls_lbl, sc_list in per_class_scores.items():
        if len(sc_list) > 0:
            per_class_ap.append(float(np.mean(sc_list)))
    proxy_map = float(np.mean(per_class_ap)) if len(per_class_ap) > 0 else 0.0

    # 4.4 Handle Output: Overlay stats and write frame
    overlay_texts = [
        f"Detections: {drawn}",
        f"mAP (proxy): {proxy_map:.3f}",
        f"Infer: {infer_time * 1000.0:.1f} ms"
    ]
    y0 = 20
    for t in overlay_texts:
        cv2.putText(frame_bgr, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        y0 += 22

    video_writer.write(frame_bgr)

# Phase 5: Cleanup
video_cap.release()
video_writer.release()

# Final Summary
if total_frames > 0:
    avg_infer_ms = (total_infer_time / total_frames) * 1000.0
else:
    avg_infer_ms = 0.0

final_per_class_ap = []
for cls_lbl, sc_list in per_class_scores.items():
    if len(sc_list) > 0:
        final_per_class_ap.append(float(np.mean(sc_list)))
final_proxy_map = float(np.mean(final_per_class_ap)) if len(final_per_class_ap) > 0 else 0.0

print("Processing complete.")
print(f"Input video: {input_path}")
print(f"Output video: {output_path}")
print(f"Frames processed: {total_frames}")
print(f"Average inference time: {avg_infer_ms:.2f} ms")
print(f"Proxy mAP (mean of per-class average confidences): {final_proxy_map:.4f}")

# Report fix for the last error
print("FIX APPLIED: Model input dimensions are now correctly parsed as integers:")
print(f" - model_in_height = int(input_shape[1]) = {model_in_height}")
print(f" - model_in_width  = int(input_shape[2]) = {model_in_width}")