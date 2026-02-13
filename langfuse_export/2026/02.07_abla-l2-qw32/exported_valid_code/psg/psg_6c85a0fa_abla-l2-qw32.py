import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Read the input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Get video properties to save processed video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(image):
    # Convert to grayscale if necessary, then resize and normalize
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
    image_normalized = np.array(image_resized / 255.0 * 255, dtype=np.uint8)
    return image_normalized

def draw_detection_boxes(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]
            
            # Convert normalized coordinates to frame coordinates
            y_min, x_min, y_max, x_max = box
            height, width, _ = frame.shape
            y_min, x_min, y_max, x_max = int(y_min * height), int(x_min * width), int(y_max * height), int(x_max * width)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the image
    processed_image = preprocess_image(frame)
    
    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(processed_image, axis=0))
    interpreter.invoke()
    
    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    
    # Draw detection boxes on the frame
    draw_detection_boxes(frame, boxes[0], classes[0], scores[0])
    
    # Write the processed frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved to", output_path)