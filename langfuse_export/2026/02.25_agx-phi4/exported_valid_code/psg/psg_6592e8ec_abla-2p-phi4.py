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
if not cap.isOpened():
    raise ValueError("Error opening video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to preprocess input image
def preprocess_image(image):
    # Resize and normalize the image
    input_shape = input_details[0]['shape']
    resized_image = cv2.resize(image, (input_shape[2], input_shape[1]))
    normalized_image = np.expand_dims(resized_image / 255.0, axis=0).astype(np.float32)
    return normalized_image

# Function to draw bounding boxes and labels
def draw_boxes(frame, detections):
    for detection in detections:
        # Extract information from the detection
        ymin, xmin, ymax, xmax = detection['bbox']
        class_id = int(detection['class_id'])
        score = float(detection['score'])

        if score >= confidence_threshold:
            label = labels[class_id]
            start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))

            # Draw rectangle and label
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
            text = f"{label}: {score:.2f}"
            cv2.putText(frame, text, (start_point[0], start_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = preprocess_image(frame)
    
    # Set tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
    interpreter.invoke()

    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence score

    # Prepare detections list
    detections = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            detection = {
                'bbox': (ymin, xmin, ymax, xmax),
                'class_id': int(classes[i]),
                'score': float(scores[i])
            }
            detections.append(detection)

    # Draw bounding boxes and labels on the frame
    draw_boxes(frame, detections)

    # Write the processed frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()