import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import os

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Setup the TFLite interpreter with EdgeTPU
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width, channels = input_shape[1], input_shape[2], input_shape[3]
input_index = input_details[0]['index']

# Read input video
cap = cv2.VideoCapture(input_path)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    # Normalize and convert to UINT8
    input_data = (input_data / 255.0).astype(np.uint8)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_index, input_data)

    # Run inference
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above threshold
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)

            cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)

    # Write the frame with detections
    out.write(frame)

cap.release()
out.release()

# Note: mAP computation is not included in this script as it requires a separate dataset of ground truth annotations.