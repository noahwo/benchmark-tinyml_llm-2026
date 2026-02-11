"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: 0b55a27f
Entry Name: 0b55_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T13:52:26.799000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_0b55.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# 1.1 Imports are already handled above.

# 1.2 Paths/Parameters are defined above.

# 1.3 Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# 1.4 Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e:
    print(f"Failed to load Edge TPU delegate: {e}")
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        raise RuntimeError("Failed to initialize the interpreter with Edge TPU delegate.")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError(f"Error opening video stream or file {input_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2.2 Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

    # 2.3 Quantization Handling
    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 4.4 Handle Output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    out.write(frame)
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()