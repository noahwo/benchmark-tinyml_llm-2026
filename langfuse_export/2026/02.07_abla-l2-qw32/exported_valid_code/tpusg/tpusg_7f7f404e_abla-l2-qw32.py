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
    raise IOError("Could not open video file.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(image):
    # Resize and normalize image to uint8 as required by the model
    input_shape = input_details[0]['shape']
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))
    normalized_image = np.array(resized_image, dtype=np.uint8)
    return normalized_image

def draw_detections(frame, boxes, classes, scores):
    for box, cls, score in zip(boxes, classes, scores):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = box
            h, w, _ = frame.shape
            (xminn, xmaxx, yminn, ymaxx) = (int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h))
            
            cv2.rectangle(frame, (xminn, yminn), (xmaxx, ymaxx), (0, 255, 0), 2)
            label = f'{labels[int(cls)]}: {score:.2f}'
            cv2.putText(frame, label, (xminn, yminn - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = preprocess_image(frame)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects

    # Draw detections on the frame
    draw_detections(frame, boxes, classes, scores)

    # Write processed frame to output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()