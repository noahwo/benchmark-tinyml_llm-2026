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
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the input video
cap = cv2.VideoCapture(input_path)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(image):
    input_shape = input_details[0]['shape']
    resized_img = cv2.resize(image, (input_shape[1], input_shape[2]))
    # Convert to uint8 as expected by the model
    input_data = np.expand_dims(resized_img, axis=0)
    return input_data

def draw_detections(frame, boxes, classes, scores):
    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            score = scores[i]

            # Get coordinates
            y_min, x_min, y_max, x_max = box

            # Convert normalized coordinates to image coordinates
            h, w, _ = frame.shape
            top_left = (int(x_min * w), int(y_min * h))
            bottom_right = (int(x_max * w), int(y_max * h))

            # Draw rectangle and label
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            label = f"{labels[class_id]}: {score:.2f}"
            cv2.putText(frame, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = preprocess_image(frame)
    
    # Set tensor to point to the preprocessed input data
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
    
    # Run inference
    interpreter.invoke()
    
    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    # Draw detections on the frame
    draw_detections(frame, boxes, classes, scores)
    
    # Write the frame with detection to the output video
    out.write(frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved to", output_path)