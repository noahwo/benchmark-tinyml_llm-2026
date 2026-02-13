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
    raise IOError("Cannot open video file")

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
    # Convert to UINT8 as required by the model
    preprocessed_frame = resized_frame.astype(np.uint8)
    return np.expand_dims(preprocessed_frame, axis=0)

# Function to draw bounding boxes and labels
def draw_detections(frame, boxes, classes, scores):
    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        if score > confidence_threshold:
            ymin = int(max(1, (box[0] * frame_height)))
            xmin = int(max(1, (box[1] * frame_width)))
            ymax = int(min(frame_height, (box[2] * frame_height)))
            xmax = int(min(frame_width, (box[3] * frame_width)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{labels[int(cls)]}: {score:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_frame = preprocess_image(frame)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    
    # Draw detections on the frame
    draw_detections(frame, boxes, classes, scores)
    
    # Write the processed frame to output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved as", output_path)