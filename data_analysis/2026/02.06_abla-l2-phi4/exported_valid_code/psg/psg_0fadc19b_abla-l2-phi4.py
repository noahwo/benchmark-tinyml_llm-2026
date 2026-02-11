import os
import cv2
import numpy as np
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

input_shape = input_details[0]['shape']
height, width, channels = input_shape[1], input_shape[2], input_shape[3]

# Open video files
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def preprocess_input(image):
    # Resize and normalize image
    image_resized = cv2.resize(image, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    input_data = input_data / 255.0  # Normalize to [0,1]
    return input_data.astype(np.uint8)  # Convert to UINT8

def postprocess_output(output_data):
    boxes = output_data[0][0]  # Bounding box coordinates
    classes = output_data[1][0]  # Class indices
    scores = output_data[2][0]  # Confidence scores
    
    detections = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            class_id = int(classes[i])
            box_coordinates = (
                int(xmin * cap.get(3)),
                int(ymin * cap.get(4)),
                int((xmax - xmin) * cap.get(3)),
                int((ymax - ymin) * cap.get(4))
            )
            detections.append((class_id, scores[i], box_coordinates))
    return detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the input
    input_data = preprocess_input(frame)
    
    # Set tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    
    detections = postprocess_output((boxes, classes, scores))
    
    # Draw bounding boxes and labels on the frame
    for class_id, score, (x, y, w, h) in detections:
        label = f"{labels[class_id]}: {score:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()