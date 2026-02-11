"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: 585da405
Entry Name: 585d_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T14:11:44.504000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters from the prompt
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_585d.mp4"
confidence_threshold   = 0.5

# Phase 1: Setup
def load_labels(label_file):
    with open(label_file, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except ValueError:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = load_labels(label_path)

# Get input shape and type
height, width, channels = input_details[0]['shape'][1:]
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
ret, frame = cap.read()

if not ret:
    raise ValueError("Unable to read from video source")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

while ret:
    # Resize and preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])

            left, right, top, bottom = int(left), int(right), int(top), int(bottom)
            cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (10, 255, 0), 2)

    # Write frame with detections to output video
    out.write(frame)
    
    # Read next frame
    ret, frame = cap.read()

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()