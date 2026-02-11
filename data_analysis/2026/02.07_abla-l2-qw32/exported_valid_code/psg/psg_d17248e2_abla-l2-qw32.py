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

# Initialize TFLite interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to save processed video with the same settings
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to preprocess the image
def preprocess_image(image):
    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to expected dimensions
    input_shape = input_details[0]['shape']
    resized_img = cv2.resize(image_rgb, (input_shape[1], input_shape[2]))
    # Normalize pixel values and convert to uint8
    normalized_img = np.array(resized_img / 255.0 * 255, dtype=np.uint8)
    # Add a batch dimension
    input_tensor = np.expand_dims(normalized_img, axis=0)
    return input_tensor

# Function to draw bounding boxes and labels on the image
def draw_detections(image, boxes, classes, scores):
    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        if score > confidence_threshold:
            # Scale the coordinates to the original size of the frame
            ymin = int(max(1, (box[0] * frame_height)))
            xmin = int(max(1, (box[1] * frame_width)))
            ymax = int(min(frame_height, (box[2] * frame_height)))
            xmax = int(min(frame_width, (box[3] * frame_width)))

            # Draw the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Get label and score text
            label = labels[int(cls)]
            score_text = f'{label}: {score:.2f}'

            # Draw the label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(score_text, font, font_scale, thickness)
            cv2.rectangle(image, (xmin, ymin - text_height - 4), (xmin + text_width, ymin), (0, 255, 0), -1)

            # Draw the label
            cv2.putText(image, score_text, (xmin, ymin - 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_tensor = preprocess_image(frame)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    # Extract results from output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Draw detections on the frame
    output_frame = draw_detections(frame, boxes, classes, scores)

    # Write the processed frame to the output video file
    out.write(output_frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()