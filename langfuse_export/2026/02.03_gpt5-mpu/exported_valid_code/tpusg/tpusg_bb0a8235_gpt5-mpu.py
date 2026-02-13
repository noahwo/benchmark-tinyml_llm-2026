import os
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports: TFLite Interpreter with EdgeTPU delegate
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    runtime_source = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
        runtime_source = "tensorflow.lite"
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to import TFLite Interpreter. Details: {e}")

# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
try:
    confidence_threshold = float('0.5')
except Exception:
    confidence_threshold = 0.5

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1.3 Load Labels
def load_labels(path):
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Handle possible "id label" or "label" formats
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        labels.append(parts[1])
                    else:
                        labels.append(line)
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to load labels from {path}. Details: {e}")
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
using_edgetpu = True
interpreter = None
delegate_error_msgs = []
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e1:
    delegate_error_msgs.append(str(e1))
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e2:
        delegate_error_msgs.append(str(e2))
        # Fallback to CPU interpreter with informative message
        using_edgetpu = False
        try:
            interpreter = Interpreter(model_path=model_path)
            print("WARNING: EdgeTPU delegate failed to load. Falling back to CPU execution.")
            print("Details (attempts):")
            for idx, msg in enumerate(delegate_error_msgs, 1):
                print(f"  Attempt {idx}: {msg}")
            print("Ensure the EdgeTPU runtime is installed: sudo apt-get install libedgetpu1-std")
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

# 2.1 Acquire Input Data: open video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit(f"ERROR: Cannot open input video: {input_path}")

# Prepare video writer with the same resolution/fps as input
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or np.isnan(fps):
    fps = 30.0  # default fallback
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
if not writer.isOpened():
    cap.release()
    raise SystemExit(f"ERROR: Cannot open output video writer: {output_path}")

# Data structures for "mAP" proxy computation (mean of average confidences per class)
per_class_confidences = {}  # class_id -> list of confidences

frame_index = 0
inference_times = []

# 2.4 Loop control: process until the end of video
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_index += 1

    # 2.2 Preprocess Data: resize, color convert, batch dimension
    # Keep a copy of original frame for drawing
    original_frame = frame_bgr.copy()
    resized_bgr = cv2.resize(frame_bgr, (input_width, input_height))
    input_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_rgb, axis=0)

    # 2.3 Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = input_data.astype(input_dtype)

    # =========================
    # Phase 3: Inference
    # =========================
    interpreter.set_tensor(input_index, input_data)
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    inference_time_ms = (t1 - t0) * 1000.0
    inference_times.append(inference_time_ms)

    # =========================
    # Phase 4: Output Interpretation & Handling
    # =========================

    # 4.1 Get Output Tensor(s)
    # The detector typically outputs: boxes [1,N,4], classes [1,N], scores [1,N], count [1]
    boxes = None
    classes = None
    scores = None
    count = None

    # Collect raw outputs
    raw_outputs = []
    for od in output_details:
        raw = interpreter.get_tensor(od['index'])
        raw_outputs.append(raw)

    # Heuristics to parse outputs
    for arr in raw_outputs:
        # boxes detection
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 4:
            boxes = arr[0]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            # Could be classes or scores
            candidate = arr[0]
            if candidate.dtype in (np.float32, np.float64):
                # Scores are in [0, 1], classes typically >1 for COCO
                if candidate.size > 0 and np.nanmax(candidate) <= 1.0 + 1e-3:
                    scores = candidate.astype(np.float32)
                else:
                    classes = candidate.astype(np.int32)
            else:
                # integer classes
                classes = candidate.astype(np.int32)
        elif arr.ndim == 1 and arr.size == 1:
            try:
                count = int(arr[0])
            except Exception:
                pass
        elif arr.ndim == 2 and arr.shape[1] == 4:
            # some models may output [N,4] directly
            boxes = arr

    if scores is None and boxes is not None:
        # If scores didn't parse but boxes exist, assume length N and set default scores of 1
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
    if classes is None and scores is not None:
        classes = np.zeros_like(scores, dtype=np.int32)
    if boxes is None:
        # No boxes returned, skip drawing and write frame as is
        writer.write(original_frame)
        continue
    if count is None:
        count = min(len(scores), boxes.shape[0]) if scores is not None else boxes.shape[0]
    count = int(count)

    # 4.2 Interpret Results: Convert raw outputs to human-readable detections
    detections = []
    # Determine if boxes are normalized [0,1] or absolute
    max_box_val = float(np.max(boxes)) if boxes.size > 0 else 0.0
    boxes_are_normalized = max_box_val <= 1.5  # heuristic threshold

    for i in range(count):
        score = float(scores[i]) if i < len(scores) else 0.0
        class_id = int(classes[i]) if i < len(classes) else 0
        y_min, x_min, y_max, x_max = boxes[i].tolist()

        # 4.3 Post-processing: confidence filtering and coordinate scaling/clipping
        if score < confidence_threshold:
            continue

        if boxes_are_normalized:
            # scale to original frame
            y_min_px = int(max(0, min(orig_height - 1, y_min * orig_height)))
            x_min_px = int(max(0, min(orig_width - 1, x_min * orig_width)))
            y_max_px = int(max(0, min(orig_height - 1, y_max * orig_height)))
            x_max_px = int(max(0, min(orig_width - 1, x_max * orig_width)))
        else:
            # if model outputs absolute coords relative to model input, scale to original
            scale_y = orig_height / float(input_height)
            scale_x = orig_width / float(input_width)
            y_min_px = int(max(0, min(orig_height - 1, y_min * scale_y)))
            x_min_px = int(max(0, min(orig_width - 1, x_min * scale_x)))
            y_max_px = int(max(0, min(orig_height - 1, y_max * scale_y)))
            x_max_px = int(max(0, min(orig_width - 1, x_max * scale_x)))

        # Ensure valid box
        if x_max_px <= x_min_px or y_max_px <= y_min_px:
            continue

        detections.append({
            'bbox': (x_min_px, y_min_px, x_max_px, y_max_px),
            'score': score,
            'class_id': class_id,
        })

    # Draw detections and maintain per-class confidences for "mAP" proxy
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        class_id = det['class_id']
        class_name = labels[class_id] if 0 <= class_id < len(labels) else f'id_{class_id}'

        # Draw rectangle
        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Put label text
        label_text = f"{class_name}: {score:.2f}"
        # Determine text position
        text_x, text_y = x1, max(0, y1 - 10)
        cv2.putText(original_frame, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(original_frame, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Update per-class confidences for mAP proxy computation
        if class_id not in per_class_confidences:
            per_class_confidences[class_id] = []
        per_class_confidences[class_id].append(score)

    # Compute a simple "mAP" proxy as mean of per-class average confidences (since no ground-truth is provided)
    ap_values = []
    for cls_id, conf_list in per_class_confidences.items():
        if len(conf_list) > 0:
            ap_values.append(float(np.mean(conf_list)))
    approx_map = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0

    # Overlay stats: mAP (proxy), frame index, inference time
    stats_text_1 = f"mAP: {approx_map:.3f}  |  Detections: {len(detections)}"
    stats_text_2 = f"Inference: {inference_time_ms:.1f} ms  |  Backend: {'EdgeTPU' if using_edgetpu else runtime_source}"
    cv2.putText(original_frame, stats_text_1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 3, cv2.LINE_AA)
    cv2.putText(original_frame, stats_text_1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(original_frame, stats_text_2, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 3, cv2.LINE_AA)
    cv2.putText(original_frame, stats_text_2, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 4.4 Handle Output: write annotated frame to output video
    writer.write(original_frame)

# 4.5 Loop Continuation handled by while loop until frames are exhausted

# =========================
# Phase 5: Cleanup
# =========================

cap.release()
writer.release()

# Final summary printout
if len(inference_times) > 0:
    print(f"Processed {frame_index} frames.")
    print(f"Average inference time: {np.mean(inference_times):.2f} ms "
          f"(min {np.min(inference_times):.2f} ms, max {np.max(inference_times):.2f} ms)")
else:
    print("No frames processed.")

print(f"Output saved to: {output_path}")