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
    labels = {i: line.strip() for i, line in enumerate(f)}

# Initialize TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file
cap = cv2.VideoCapture(input_path)

# Prepare video writer for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to preprocess the frame
def preprocess_frame(frame):
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = np.expand_dims(resized_frame.astype(np.uint8), axis=0)
    return normalized_frame

# Inference and output handling
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    preprocessed_frame = preprocess_frame(frame)
    
    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    
    # Run inference
    interpreter.invoke()
    
    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
    
    # Draw detection boxes and labels
    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()