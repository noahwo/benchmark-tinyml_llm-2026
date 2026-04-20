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

# Read the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(image):
    # Resize and normalize image
    input_shape = input_details[0]['shape']
    resized_img = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_img, axis=0)
    input_data = input_data.astype(np.uint8)  # Ensure the data type is UINT8
    return input_data

def draw_detection_boxes(frame, boxes, classes, scores):
    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = box
            h, w, _ = frame.shape
            (xminn, xmaxx, yminn, ymaxx) = (int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h))
            cv2.rectangle(frame, (xminn, yminn), (xmaxx, ymaxx), (10, 255, 0), 2)
            label = f"{labels[int(cls)]}: {score:.2f}"
            cv2.putText(frame, label, (xminn, yminn - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    draw_detection_boxes(frame, boxes, classes, scores)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()