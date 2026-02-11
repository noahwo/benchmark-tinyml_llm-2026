import os
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Setup TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input shape and data type
input_shape = input_details[0]['shape']
input_data_type = input_details[0]['dtype']

# Read video
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError("Error opening video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing: resize and normalize input image
    input_image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_image = np.expand_dims(input_image, axis=0)
    
    # Convert to UINT8
    input_image = np.uint8(input_image)

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)

            label = labels[int(classes[i])]
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Output handling
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()