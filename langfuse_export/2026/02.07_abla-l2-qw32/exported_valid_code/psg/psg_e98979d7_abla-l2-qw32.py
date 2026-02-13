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

# Initialize TFLite interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to preprocess the image
def preprocess_image(frame):
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0)
    # Convert to UINT8 as required by the model
    input_data = np.uint8(input_data * 255.0)
    return input_data

# Function to draw bounding boxes on the frame
def draw_boxes(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            height, width, _ = frame.shape
            (xminn, xmaxx, yminn, ymaxx) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(frame, (int(xminn), int(yminn)), (int(xmaxx), int(ymaxx)), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(xminn), int(yminn) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Main loop to process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    preprocessed_frame = preprocess_image(frame)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

    # Run inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw bounding boxes on the frame
    draw_boxes(frame, boxes, classes, scores)

    # Write the frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()