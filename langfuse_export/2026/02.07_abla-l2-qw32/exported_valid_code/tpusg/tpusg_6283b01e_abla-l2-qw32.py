import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
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
    raise IOError("Cannot open video")

# Video writer setup
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def preprocess(frame):
    # Convert frame to RGB and resize
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    return np.expand_dims(resized_frame, axis=0).astype(np.uint8)

def draw_detection_boxes(image, boxes, classes, scores):
    for box, cls, score in zip(boxes, classes, scores):
        if score > confidence_threshold:
            y_min = int(max(1, box[0] * image.shape[0]))
            x_min = int(max(1, box[1] * image.shape[1]))
            y_max = int(min(image.shape[0], box[2] * image.shape[0]))
            x_max = int(min(image.shape[1], box[3] * image.shape[1]))

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f'{labels[int(cls)]}: {score:.2f}'
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_data = preprocess(frame)
    
    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw detection boxes and labels on the frame
    draw_detection_boxes(frame, boxes, classes, scores)

    # Write processed frame to output video file
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed.")