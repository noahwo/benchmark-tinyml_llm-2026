import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import os

# Configuration parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter with EdgeTPU acceleration
try:
    interpreter = Interpreter(model_path=model_path, 
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    try:
        interpreter = Interpreter(model_path=model_path, 
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e2:
        raise RuntimeError("Failed to load EdgeTPU delegate") from e2

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = (input_details[0]['dtype'] == np.float32)

# Import OpenCV for video processing
import cv2

# Read video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Interpret results
    for i in range(num_detections):
        if scores[0][i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = labels[int(classes[0][i])] + ": " + str(scores[0][i])
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write processed frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()