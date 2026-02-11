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

# Load labels
def load_labels(path):
    with open(path, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Initialize the interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

# Video capture and output setup
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file {}".format(input_path))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Preprocess the frame
def preprocess_frame(frame):
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.float32(input_data - 127.5) / 127.5
    return input_data

# Perform inference on the frame
def infer(interpreter, frame):
    input_data = preprocess_frame(frame)

    # Ensure the input data type matches what's expected by the model
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence score

    return boxes, classes, scores

# Process video frames and draw bounding boxes with labels
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    boxes, classes, scores = infer(interpreter, frame)
    
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            
            label = labels[int(classes[i])]
            text = f"{label}: {int(scores[i] * 100)}%"
            cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)
    
    out.write(frame)

cap.release()
out.release()

# Note: Mean Average Precision (mAP) computation is a complex task and typically involves
# comparing the predicted bounding boxes with ground truth data, which requires additional setup.