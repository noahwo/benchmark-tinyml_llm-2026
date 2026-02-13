import os
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import cv2

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from the label map
def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

labels = load_labels(label_path)

# Setup: Initialize TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']
box_output_index = output_details[0]['index']
class_output_index = output_details[1]['index']

# Preprocessing: Read video
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Helper functions
def set_input_tensor(interpreter, image):
    tensor_index = input_details[0]['index']
    input_shape = input_details[0]['shape']
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image, axis=0)
    interpreter.set_tensor(tensor_index, input_data)

def get_output(interpreter):
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    return boxes, classes, scores

# Inference loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    set_input_tensor(interpreter, frame)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Get the results
    boxes, classes, scores = get_output(interpreter)

    # Draw bounding boxes and labels on the frame
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = f"{object_name}: {int(scores[i]*100)}%"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5 * label_size[1]))),
                          (int(left + round(1.5 * label_size[0])), int(top + base_line)),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    out.write(frame)

# Release resources
cap.release()
out.release()

# Placeholder for mAP calculation (requires additional implementation)
print("mAP calculation is not implemented in this script.")