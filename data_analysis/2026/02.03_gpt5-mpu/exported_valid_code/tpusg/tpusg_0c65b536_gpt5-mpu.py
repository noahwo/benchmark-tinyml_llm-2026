import os
import sys
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports: Try tflite_runtime first, fallback to tensorflow.lite
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_backend = 'tflite_runtime'
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        tflite_backend = 'tensorflow.lite'
    except Exception as e:
        print("Error: Failed to import TFLite Interpreter. Ensure 'tflite-runtime' or 'tensorflow' is installed.")
        sys.exit(1)

# OpenCV is needed for video I/O and drawing
import cv2

# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3 Load Labels (if provided and relevant)
def load_labels(path):
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                name = line.strip()
                if len(name) > 0:
                    labels.append(name)
    except Exception as e:
        print(f"Warning: Failed to load labels from {path}: {e}")
        labels = []
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
interpreter = None
delegate_errors = []
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e1:
    delegate_errors.append(str(e1))
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e2:
        delegate_errors.append(str(e2))
        print("Error: Failed to load EdgeTPU delegate for the model.")
        print("Tried delegates: 'libedgetpu.so.1.0' and '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'.")
        print("Delegate load errors:")
        for idx, msg in enumerate(delegate_errors, 1):
            print(f"  {idx}. {msg}")
        print("Ensure the EdgeTPU runtime is installed and the EdgeTPU is connected.")
        sys.exit(1)

# Allocate tensors
try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error: Failed to allocate tensors: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    print("Error: Model has no input tensors.")
    sys.exit(1)

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_height, input_width = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Attempt to map output indices (typical SSD order)
if len(output_details) < 4:
    print("Error: Unexpected number of output tensors. Expected 4 for SSD-style detection model.")
    sys.exit(1)

boxes_idx = output_details[0]['index']
classes_idx = output_details[1]['index']
scores_idx = output_details[2]['index']
num_idx = output_details[3]['index']

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1 Acquire Input Data - open video file
if not os.path.exists(input_path):
    print(f"Error: Input video not found: {input_path}")
    sys.exit(1)

video_capture = cv2.VideoCapture(input_path)
if not video_capture.isOpened():
    print(f"Error: Failed to open input video: {input_path}")
    sys.exit(1)

# Prepare VideoWriter for output
fps = video_capture.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0  # Fallback if FPS is not detected
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
if not video_writer.isOpened():
    print(f"Error: Failed to open output video for writing: {output_path}")
    video_capture.release()
    sys.exit(1)

# For mAP-like aggregation (proxy in absence of ground-truth): per-class average of detection confidences
per_class_scores = {}  # class_id -> list of scores
total_frames_processed = 0

# Utility: preprocess frame to model input
def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Ensure dtype matches model requirements (usually uint8 for EdgeTPU models)
        input_data = np.asarray(input_data, dtype=input_dtype)
    return input_data

# Utility: compute running mAP-like metric from per_class_scores
def compute_running_map(per_class_scores_dict):
    if not per_class_scores_dict:
        return 0.0
    class_avgs = []
    for cls_id, scores in per_class_scores_dict.items():
        if len(scores) > 0:
            class_avgs.append(float(np.mean(scores)))
    if len(class_avgs) == 0:
        return 0.0
    return float(np.mean(class_avgs))

# Detection drawing parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = max(0.5, min(frame_width, frame_height) / 1000.0)
thickness = max(1, int(min(frame_width, frame_height) / 500))

# Main processing loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    total_frames_processed += 1

    # 2.2 Preprocess Data
    input_data = preprocess_frame(frame)

    # Phase 3: Inference
    # 3.1 Set Input Tensor
    interpreter.set_tensor(input_index, input_data)
    # 3.2 Run Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # 4.1 Get Output Tensors
    boxes = interpreter.get_tensor(boxes_idx)[0]       # [N, 4] normalized
    classes = interpreter.get_tensor(classes_idx)[0]   # [N]
    scores = interpreter.get_tensor(scores_idx)[0]     # [N]
    num = int(interpreter.get_tensor(num_idx)[0])      # scalar

    # 4.2 Interpret Results
    # Convert normalized boxes to pixel coordinates and map class indices to labels
    detections = []
    for i in range(num):
        score = float(scores[i])
        if score < confidence_threshold:
            continue
        cls_id = int(classes[i])
        y_min, x_min, y_max, x_max = boxes[i]

        # 4.3 Post-processing: scale coords to frame size and clip to valid ranges
        x_min_px = int(max(0, min(frame_width - 1, x_min * frame_width)))
        y_min_px = int(max(0, min(frame_height - 1, y_min * frame_height)))
        x_max_px = int(max(0, min(frame_width - 1, x_max * frame_width)))
        y_max_px = int(max(0, min(frame_height - 1, y_max * frame_height)))

        # Ensure proper box ordering
        x1, y1 = min(x_min_px, x_max_px), min(y_min_px, y_max_px)
        x2, y2 = max(x_min_px, x_max_px), max(y_min_px, y_max_px)

        label = str(cls_id)
        if labels and 0 <= cls_id < len(labels):
            label = labels[cls_id]

        detections.append({
            'box': (x1, y1, x2, y2),
            'class_id': cls_id,
            'label': label,
            'score': score
        })

        # Accumulate for mAP-like metric
        if cls_id not in per_class_scores:
            per_class_scores[cls_id] = []
        per_class_scores[cls_id].append(score)

    # 4.4 Handle Output: Draw detections and running mAP on the frame
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label_text = f"{det['label']} {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        # Background for text for visibility
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_bg_x2 = min(frame_width - 1, x1 + text_w + 4)
        text_bg_y2 = min(frame_height - 1, y1 + text_h + 4)
        cv2.rectangle(frame, (x1, y1), (text_bg_x2, text_bg_y2), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 + text_h + 1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    running_map = compute_running_map(per_class_scores)
    map_text = f"mAP: {running_map:.3f}"
    cv2.putText(frame, map_text, (10, 30), font, font_scale * 1.2, (0, 0, 255), max(1, thickness), cv2.LINE_AA)

    # Write the annotated frame to output video
    video_writer.write(frame)

# Phase 5: Cleanup
video_capture.release()
video_writer.release()

# Print summary
final_map = compute_running_map(per_class_scores)
print("Processing completed.")
print(f"Total frames processed: {total_frames_processed}")
print(f"Output saved to: {output_path}")
print(f"Calculated mAP: {final_map:.4f}")