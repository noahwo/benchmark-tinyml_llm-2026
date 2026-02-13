import os
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Setup

# Paths/Parameters
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels
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
        raise e

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop

import cv2

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {input_path}")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    scores = interpreter.get_tensor(output_details[3]['index'])[0]

    # Check if scores is a single float (no detections) or an array
    if isinstance(scores, np.float32):
        num_detections = 0
    else:
        num_detections = len(scores)

    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            ymin = int(max(1, boxes[i][0] * frame_height))
            xmin = int(max(1, boxes[i][1] * frame_width))
            ymax = int(min(frame_height, boxes[i][2] * frame_height))
            xmax = int(min(frame_width, boxes[i][3] * frame_width))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()