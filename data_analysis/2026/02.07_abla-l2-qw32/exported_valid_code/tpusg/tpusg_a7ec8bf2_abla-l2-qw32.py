import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Initialize the TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Video writer setup
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Function to preprocess the image
def preprocess_image(frame):
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)  # Ensure the data type is uint8
    return input_data

# Function to draw bounding boxes and labels
def draw_detections(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]

            # Draw rectangle
            top_left = (int(box[1] * width), int(box[0] * height))
            bottom_right = (int(box[3] * width), int(box[2] * height))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Draw label
            label_text = f"{label}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_origin = (top_left[0], top_left[1] - 10 if top_left[1] > 20 else top_left[1] + 10)
            cv2.rectangle(frame, (label_origin[0] - 5, label_origin[1] + 5), 
                          (label_origin[0] + label_size[0] + 5, label_origin[1] - label_size[1] - 5), 
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label_text, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    draw_detections(frame, boxes[0], classes[0], scores[0])

    out.write(frame)

cap.release()
out.release()