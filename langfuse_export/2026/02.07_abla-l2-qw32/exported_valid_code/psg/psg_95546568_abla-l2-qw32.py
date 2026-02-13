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

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Could not open video file {input_path}")

# Get video properties to initialize the video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)  # Ensure the data type is uint8

    # Set tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Draw detection boxes with labels
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            box = boxes[0][i]
            class_id = int(classes[0][i])
            label = labels[class_id]
            score = float(scores[0][i])

            # Scale the bounding box coordinates
            ymin, xmin, ymax, xmax = box
            h, w, _ = frame.shape
            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)

            # Draw rectangle and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (int(left), int(top) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detection results to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()