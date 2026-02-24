import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Video capture and writer setup
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to preprocess input image
def preprocess_image(frame):
    # Resize and normalize the image
    input_size = input_details[0]['shape'][1:3]
    frame_resized = cv2.resize(frame, input_size)
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32)
    input_data = (input_data / 127.5) - 1.0
    return input_data

# Function to draw bounding boxes and labels on the frame
def draw_boxes(frame, detections):
    for detection in detections:
        # Extract information from the detection
        ymin, xmin, ymax, xmax, score, class_id = detection
        if score < confidence_threshold:
            continue
        
        # Scale box to original image size
        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                     ymin * frame.shape[0], ymax * frame.shape[0])
        
        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        label = f"{labels[int(class_id)]}: {score:.2f}"
        cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_image(frame)
    
    # Set tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Prepare detections for drawing
    detections = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            class_id = int(classes[i])
            score = float(scores[i])
            detections.append((ymin, xmin, ymax, xmax, score, class_id))

    # Draw bounding boxes and labels on the frame
    draw_boxes(frame, detections)

    # Write the processed frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()