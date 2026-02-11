import numpy as np
import time
from tflite_runtime.interpreter import Interpreter, load_delegate
import os
import cv2

# Phase 1: Setup

# Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load EdgeTPU delegate: {e}")
        raise

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)
height, width = input_details[0]['shape'][1:3]

# Phase 2: Input Acquisition & Preprocessing Loop

# Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Interpret Results
    for i in range(num_detections):
        if scores[0][i] > confidence_threshold:
            class_id = int(classes[0][i])
            label = labels[class_id]
            box = boxes[0][i]

            # Post-processing: Apply confidence thresholding, coordinate scaling, and bounding box clipping
            ymin = int(max(1, (box[0] * frame.shape[0])))
            xmin = int(max(1, (box[1] * frame.shape[1])))
            ymax = int(min(frame.shape[0], (box[2] * frame.shape[0])))
            xmax = int(min(frame.shape[1], (box[3] * frame.shape[1])))

            # Handle Output: Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {int(scores[0][i] * 100)}%', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write processed frame to output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()