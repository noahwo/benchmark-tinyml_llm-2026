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
    input_data = np.expand_dims(resized_img, axis=0)
    input_data = input_data.astype(np.uint8)  # Ensure the data type is UINT8
    return input_data

def draw_detection_boxes(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]

            # Convert normalized coordinates to pixel values
            top_left = (int(box[0] * frame_width), int(box[1] * frame_height))
            bottom_right = (int(box[2] * frame_width), int(box[3] * frame_height))

            # Draw rectangle and label text
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Process each frame
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
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw detection boxes on the frame
    draw_detection_boxes(frame, boxes, classes, scores)

    # Write the processed frame to output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()