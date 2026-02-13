import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Could not open video file: {input_path}")

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(frame):
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def postprocess_detections(output_data, frame):
    boxes = output_data['detection_boxes'][0]
    classes = output_data['detection_classes'][0].astype(int)
    scores = output_data['detection_scores'][0]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            label = labels[classes[i]]
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {scores[i]:.2f}", (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Main loop for processing video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Get the results
    output_data = {
        'detection_boxes': interpreter.get_tensor(output_details[0]['index']),
        'detection_classes': interpreter.get_tensor(output_details[1]['index']),
        'detection_scores': interpreter.get_tensor(output_details[2]['index'])
    }

    frame_with_detections = postprocess_detections(output_data, frame)
    out.write(frame_with_detections)

cap.release()
out.release()
cv2.destroyAllWindows()