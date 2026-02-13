import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {input_path}")

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input data
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)  # Ensure the input is UINT8

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw detection boxes and labels
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = map(int, (boxes[i] * np.array([frame_height, frame_width, frame_height, frame_width])))
            label = labels[int(classes[i])]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {scores[i]:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()