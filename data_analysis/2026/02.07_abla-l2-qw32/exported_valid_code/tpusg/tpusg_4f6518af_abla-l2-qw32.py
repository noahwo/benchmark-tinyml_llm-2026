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
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(label_path)

# Setup TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the image
def preprocess_image(frame):
    height, width = frame.shape[:2]
    img_resized = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(img_resized, axis=0)
    return input_data

# Function to draw bounding boxes and labels
def draw_boxes(frame, boxes, classes, scores):
    for box, cls, score in zip(boxes, classes, scores):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = box
            height, width = frame.shape[:2]
            (xminn, xmaxx, ymine, ymaxe) = (int(xmin * width), int(xmax * width),
                                            int(ymin * height), int(ymax * height))
            cv2.rectangle(frame, (xminn, ymine), (xmaxx, ymaxe), (10, 255, 0), 2)
            label = f'{labels[int(cls)]}: {score:.2f}'
            cv2.putText(frame, label, (xminn, ymine - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Main processing loop
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    draw_boxes(frame, boxes, classes, scores)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()