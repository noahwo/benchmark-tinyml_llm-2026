"""
Extracted from: raw_qwen2.5-coder:32b_1974_tpusg_batch
Entry ID: bf2bc90f
Entry Name: bf2b_tpu_sketch_generator
Session ID: qwen2.5-coder:32b_1974_tpusg_batch
Timestamp: 2025-07-30T05:57:14.377000+00:00
Tags: benchmark, qwen2.5-coder:32b, tpu_sketch_generator
"""

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import cv2

# Phase 1: Setup
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_bf2b.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load libedgetpu.so.1.0: {e}")
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0: {e}")
        raise

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    for i in range(num_detections):
        confidence = float(scores[0][i])
        if confidence > confidence_threshold:
            class_id = int(classes[0][i])
            label = labels[class_id]
            box = boxes[0][i] * np.array([frame_height, frame_width, frame_height, frame_width])

            y_min, x_min, y_max, x_max = box.astype(int)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)
            label_text = f'{label}: {confidence:.2f}'
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()