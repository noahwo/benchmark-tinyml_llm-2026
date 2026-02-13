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

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file
cap = cv2.VideoCapture(input_path)

# Prepare to write the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess(frame):
    # Resize the frame to the input size expected by the model
    resized_frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    # Convert the image to grayscale if needed and normalize to UINT8
    preprocessed_frame = np.array(resized_frame).astype(np.uint8)
    return preprocessed_frame

def draw_detections(frame, boxes, classes, scores):
    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            score = scores[i]
            label = labels[class_id]

            # Convert normalized coordinates to pixel values
            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * frame_height)
            xmin = int(xmin * frame_width)
            ymax = int(ymax * frame_height)
            xmax = int(xmax * frame_width)

            # Draw the bounding box and label on the image
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess(frame)
    input_tensor = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run the inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Draw detections on the frame
    draw_detections(frame, boxes[0], classes[0], scores[0])

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved to", output_path)