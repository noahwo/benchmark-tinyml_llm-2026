import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Could not open video file {input_path}")

# Video writer setup for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Preprocessing function
def preprocess_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    return input_data

# Function to draw bounding boxes and labels
def draw_boxes(frame, boxes, classes, scores):
    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            score = scores[i]
            label = f"{labels[class_id]}: {score:.2f}"
            top_left = (int(box[0] * width), int(box[1] * height))
            bottom_right = (int(box[2] * width), int(box[3] * height))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Main loop for video processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
    interpreter.invoke()

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    draw_boxes(frame, boxes, classes, scores)
    out.write(frame)

cap.release()
out.release()