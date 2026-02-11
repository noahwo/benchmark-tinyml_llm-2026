import cv2
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter with Edge TPU delegate
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Video writer for saving the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def draw_boxes(frame, boxes, class_ids, confidences):
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        if confidence >= confidence_threshold:
            ymin, xmin, ymax, xmax = box
            label = labels[int(class_id)]
            color = (10, 255, 0)
            cv2.rectangle(frame, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (int(xmin * width), int(ymin * height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw bounding boxes and labels on the frame
    draw_boxes(frame, boxes, class_ids, scores)

    # Write the frame with detection to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()