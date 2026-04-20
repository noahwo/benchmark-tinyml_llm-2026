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

# Prepare video writer for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(image):
    # Resize and normalize image
    input_shape = input_details[0]['shape']
    resized_img = cv2.resize(image, (input_shape[1], input_shape[2]))
    normalized_img = np.array(resized_img, dtype=np.uint8)
    return normalized_img

# Function to draw bounding boxes on the frame
def draw_boxes(frame, boxes, classes, scores):
    for box, cls, score in zip(boxes, classes, scores):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = box
            h, w, _ = frame.shape
            (xminn, xmaxx, yminn, ymaxx) = (int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h))
            cv2.rectangle(frame, (xminn, yminn), (xmaxx, ymaxx), (10, 255, 0), 2)
            label = f"{labels[int(cls)]}: {score:.2f}"
            cv2.putText(frame, label, (xminn, yminn - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = preprocess_image(frame)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Set tensor to point to the preprocessed data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw bounding boxes on the frame
    draw_boxes(frame, boxes, classes, scores)

    # Write the frame with detection to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()