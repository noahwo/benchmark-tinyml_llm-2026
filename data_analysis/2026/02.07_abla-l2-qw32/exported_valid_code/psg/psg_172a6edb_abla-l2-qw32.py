import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Initialize the TFLite interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Video writer setup for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(frame):
    # Resize and convert to uint8
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    preprocessed_frame = np.expand_dims(resized_frame, axis=0)
    preprocessed_frame = preprocessed_frame.astype(np.uint8)  # Convert to uint8
    return preprocessed_frame

def postprocess_output(detection_boxes, detection_classes, detection_scores):
    height, width, _ = frame.shape
    results = []
    for box, score, class_id in zip(detection_boxes[0], detection_scores[0], detection_classes[0]):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            results.append((left, top, right - left, bottom - top, labels[int(class_id)], score))
    return results

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Set tensor to interpreter and run inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    interpreter.invoke()

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Postprocess the results
    detections = postprocess_output(boxes, classes, scores)

    # Draw detection boxes and labels on the frame
    for (x, y, w, h, label, score) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Write the frame with detections to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()