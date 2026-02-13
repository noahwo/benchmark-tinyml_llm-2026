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
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Setup TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

def preprocess_frame(frame):
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0)
    # Convert to UINT8
    input_data = np.uint8(input_data)
    return input_data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    input_data = preprocess_frame(frame)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))
    
    for i in range(num_detections):
        if scores[0][i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            label = labels[int(classes[0][i])]
            
            # Convert normalized coordinates to frame dimensions
            (imH, imW) = frame.shape[:2]
            (xminn, xmaxx, yminn, ymaxx) = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
            
            cv2.rectangle(frame, (int(xminn), int(yminn)), (int(xmaxx), int(ymaxx)), (10, 255, 0), 2)
            label_text = f"{label}: {scores[0][i]:.2f}"
            cv2.putText(frame, label_text, (int(xminn), int(yminn) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()