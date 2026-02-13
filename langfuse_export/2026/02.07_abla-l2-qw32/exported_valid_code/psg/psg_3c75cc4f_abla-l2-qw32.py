import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Setup TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the input video file
cap = cv2.VideoCapture(input_path)

# Prepare to write the output video file
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing: Convert to grayscale and resize
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int64)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw detection boxes with labels and confidence scores on the frame
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = map(int, (boxes[i] * np.array([frame_height, frame_width, frame_height, frame_width])))
            label = labels[classes[i]]
            score = float(scores[i])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()