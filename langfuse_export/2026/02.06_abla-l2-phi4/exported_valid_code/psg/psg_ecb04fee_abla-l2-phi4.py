import os
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from label map
def load_labels(path):
    with open(path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Setup TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# Load labels
labels = load_labels(label_path)

# Open video capture
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Function to draw bounding boxes on the image
def draw_boxes(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),
                          (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing: Resize and normalize the input frame
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Set model input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence score

    # Draw bounding boxes on the frame
    draw_boxes(frame, boxes, classes, scores)

    # Write the frame with detections to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()

print("Object detection completed and saved at:", output_path)