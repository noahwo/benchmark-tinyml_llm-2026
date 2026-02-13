import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Setup TFLite interpreter with EdgeTPU
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def preprocess_image(image):
    # Resize image to input tensor size
    input_shape = input_details[0]['shape']
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_image, axis=0)
    return input_data

def postprocess_output(output_data):
    # Extract bounding boxes and scores
    detection_boxes = output_data[0][0]
    detection_scores = output_data[1][0]
    num_detections = int(output_data[3][0])
    detections = []
    for i in range(num_detections):
        score = float(detection_scores[i])
        if score > confidence_threshold:
            bbox = detection_boxes[i]
            label_id = int(output_data[2][0][i])
            label = labels[label_id]
            detections.append((bbox, score, label))
    return detections

# mAP calculation placeholder (not fully implemented here)
def calculate_mAP(gt_boxes, pred_boxes):
    # Placeholder function for demonstration
    return 0.0

frame_count = 0
all_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    input_data = preprocess_image(frame)
    
    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get output tensors
    output_data = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]
    
    # Postprocess the output data
    detections = postprocess_output(output_data)
    all_detections.append(detections)
    
    # Draw detection boxes and labels on the frame
    for (bbox, score, label) in detections:
        ymin, xmin, ymax, xmax = bbox
        h, w, _ = frame.shape
        left = int(xmin * w)
        top = int(ymin * h)
        right = int(xmax * w)
        bottom = int(ymax * h)
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Write the frame to output video
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# Placeholder for mAP calculation
mAP = calculate_mAP([], all_detections)  # Provide ground truth boxes as needed
print(f"Mean Average Precision (mAP): {mAP:.2f}")