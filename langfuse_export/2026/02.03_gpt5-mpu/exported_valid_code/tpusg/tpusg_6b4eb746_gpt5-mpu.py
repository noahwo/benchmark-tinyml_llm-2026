import os
import sys
import time
import numpy as np
import cv2

# ==============================
# Phase 1: Setup
# ==============================

# 1.1 Imports: Interpreter and EdgeTPU delegate with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
    except Exception as e:
        print("ERROR: Failed to import TFLite Interpreter. Ensure tflite-runtime or TensorFlow Lite is installed.")
        print(f"Details: {e}")
        sys.exit(1)

# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1.3 Load Labels
def load_labels(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label = line.strip()
                if label != '':
                    labels.append(label)
    except Exception as e:
        print(f"WARNING: Failed to load labels from {label_file_path}. Details: {e}")
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
interpreter = None
delegate_error_msgs = []
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e1:
    delegate_error_msgs.append(str(e1))
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e2:
        delegate_error_msgs.append(str(e2))
        print("ERROR: Failed to load EdgeTPU delegate.")
        print("Tried the following delegate libraries:")
        print(" - libedgetpu.so.1.0")
        print(" - /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0")
        print("Original errors:")
        for idx, msg in enumerate(delegate_error_msgs, 1):
            print(f"  ({idx}) {msg}")
        print("Please ensure the Coral EdgeTPU runtime is installed on the device.")
        sys.exit(1)

try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"ERROR: Failed to allocate tensors for the TFLite interpreter. Details: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    print("ERROR: Model does not have any input tensors.")
    sys.exit(1)

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Expect input tensor shape [1, height, width, 3]
if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[3] != 3:
    print(f"ERROR: Unexpected input tensor shape: {input_shape}. Expected [1, height, width, 3].")
    sys.exit(1)

in_height, in_width = int(input_shape[1]), int(input_shape[2])

# ==============================
# Phase 2: Input Acquisition & Preprocessing Loop
# ==============================

# 2.1 Acquire Input Data: Open video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"ERROR: Failed to open input video: {input_path}")
    sys.exit(1)

# Prepare output video writer
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0  # Fallback if FPS is unavailable

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
if not out_writer.isOpened():
    print(f"ERROR: Failed to open output video for writing: {output_path}")
    cap.release()
    sys.exit(1)

# For proxy mAP calculation: accumulate per-class confidences
per_class_confidences = {}  # class_id -> list of confidences

# ==============================
# Processing Loop
# ==============================
frame_index = 0
inference_times = []

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_index += 1
    frame_h, frame_w = frame_bgr.shape[:2]

    # 2.2 Preprocess Data
    # Convert BGR to RGB, resize to model input size
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(frame_rgb, (in_width, in_height))
    input_data = np.expand_dims(resized_rgb, axis=0)

    # 2.3 Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = input_data.astype(input_dtype, copy=False)

    # ==============================
    # Phase 3: Inference
    # ==============================
    try:
        # 3.1 Set Input Tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 3.2 Run Inference
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        inference_times.append(t1 - t0)
    except Exception as e:
        print(f"ERROR: Inference failed at frame {frame_index}. Details: {e}")
        break

    # ==============================
    # Phase 4: Output Interpretation & Handling
    # ==============================

    # 4.1 Get Output Tensor(s)
    # Typical detection model outputs: boxes [1, N, 4], classes [1, N], scores [1, N], num_detections [1]
    boxes = None
    classes = None
    scores = None
    num_detections = None

    try:
        for od in output_details:
            out_tensor = interpreter.get_tensor(od['index'])
            shp = out_tensor.shape
            # Identify tensors by shape/dtype heuristics
            if len(shp) == 3 and shp[-1] == 4:
                boxes = out_tensor  # [1, N, 4]
            elif len(shp) == 2:
                # Could be classes or scores
                if out_tensor.dtype == np.float32 and np.max(out_tensor) <= 1.0 and np.min(out_tensor) >= 0.0:
                    scores = out_tensor  # [1, N] scores in [0,1]
                else:
                    classes = out_tensor  # [1, N] class indices (float or int)
            elif len(shp) == 1 and shp[0] == 1:
                num_detections = int(np.squeeze(out_tensor).astype(np.int32))
    except Exception as e:
        print(f"ERROR: Failed to retrieve output tensors at frame {frame_index}. Details: {e}")
        break

    if boxes is None or scores is None or classes is None:
        print(f"ERROR: Missing one or more output tensors (boxes/scores/classes) at frame {frame_index}.")
        break

    if num_detections is None:
        # Fallback to number of entries in scores
        num_detections = scores.shape[1] if len(scores.shape) == 2 else scores.shape[0]

    # 4.2 Interpret Results
    # Prepare lists for filtered detections
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = []

    # Flatten batch dimension
    boxes_ = boxes[0] if len(boxes.shape) == 3 else boxes
    scores_ = scores[0] if len(scores.shape) == 2 else scores
    classes_ = classes[0] if len(classes.shape) == 2 else classes

    # 4.3 Post-processing: Apply confidence threshold, scale/clip coordinates
    for i in range(int(num_detections)):
        score = float(scores_[i])
        if score < confidence_threshold:
            continue

        # Class ID handling (classes often a float array)
        class_id = int(classes_[i])

        # Bounding box in normalized ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = boxes_[i].tolist()

        # Scale to original frame size
        x1 = int(max(0, min(1.0, xmin)) * frame_w)
        y1 = int(max(0, min(1.0, ymin)) * frame_h)
        x2 = int(max(0, min(1.0, xmax)) * frame_w)
        y2 = int(max(0, min(1.0, ymax)) * frame_h)

        # Clip to frame boundaries
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y2 = max(0, min(y2, frame_h - 1))

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        filtered_boxes.append((x1, y1, x2, y2))
        filtered_classes.append(class_id)
        filtered_scores.append(score)

        # Accumulate confidences for proxy mAP per class
        if class_id not in per_class_confidences:
            per_class_confidences[class_id] = []
        per_class_confidences[class_id].append(score)

    # Drawing detections on the frame
    for (x1, y1, x2, y2), class_id, score in zip(filtered_boxes, filtered_classes, filtered_scores):
        color = (0, 255, 0)  # Green box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Determine label text
        if 0 <= class_id < len(labels):
            class_name = labels[class_id]
        else:
            class_name = f"id:{class_id}"
        label_text = f"{class_name}: {score:.2f}"

        # Draw label background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_x, text_y = x1, max(0, y1 - 5)
        box_coords = ((text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline))
        cv2.rectangle(frame_bgr, box_coords[0], box_coords[1], color, cv2.FILLED)
        cv2.putText(frame_bgr, label_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Compute a proxy mAP (since no ground-truth is provided): mean of per-class average confidences
    if len(per_class_confidences) > 0:
        class_means = [float(np.mean(conf_list)) for conf_list in per_class_confidences.values() if len(conf_list) > 0]
        mAP_proxy = float(np.mean(class_means)) if len(class_means) > 0 else 0.0
    else:
        mAP_proxy = 0.0

    # Overlay proxy mAP on frame
    map_text = f"mAP: {mAP_proxy:.3f}"
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2, cv2.LINE_AA)

    # Optionally overlay FPS (inference)
    if len(inference_times) > 0:
        avg_inf_ms = (sum(inference_times[-30:]) / min(len(inference_times), 30)) * 1000.0
        fps_text = f"Infer: {avg_inf_ms:.1f} ms"
        cv2.putText(frame_bgr, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)

    # 4.4 Handle Output: write annotated frame to output video
    out_writer.write(frame_bgr)

# 4.5 Loop ends

# ==============================
# Phase 5: Cleanup
# ==============================
cap.release()
out_writer.release()

# Print summary
total_frames = frame_index
avg_infer_ms = (sum(inference_times) / len(inference_times) * 1000.0) if len(inference_times) > 0 else 0.0
print("Processing completed.")
print(f"Input video: {input_path}")
print(f"Output video: {output_path}")
print(f"Total frames processed: {total_frames}")
print(f"Average inference time per frame: {avg_infer_ms:.2f} ms")