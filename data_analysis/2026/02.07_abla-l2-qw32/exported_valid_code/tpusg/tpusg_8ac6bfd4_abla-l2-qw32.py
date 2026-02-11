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
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file from the given path
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Get video properties to save the result video with the same settings
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(image):
    # Resize image to match input tensor shape
    input_shape = input_details[0]['shape']
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))
    
    # Normalize the image if required by the model
    mean = np.array([127.5])
    std = np.array([127.5])
    normalized_image = (resized_image.astype(np.float32) - mean) / std
    
    # Convert to UINT8 as expected by the model
    input_data = normalized_image.astype(np.uint8)
    
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def draw_detections(frame, boxes, classes, scores):
    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = box
            h, w, _ = frame.shape
            (left, right, top, bottom) = (xmin * w, xmax * w,
                                          ymin * h, ymax * h)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Draw label
            object_name = labels[int(cls)]
            label = f'{object_name}: {score:.2f}'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(top - labelSize[1], 1)
            cv2.rectangle(frame, (int(left), int(label_ymin - labelSize[1])),
                          (int(left + labelSize[0]), int(label_ymin + baseLine)), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(label_ymin)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the image
    input_data = preprocess_image(frame)
    
    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    
    # Draw detections on the frame
    draw_detections(frame, boxes, classes, scores)
    
    # Write the processed frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()