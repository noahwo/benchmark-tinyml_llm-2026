import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(filename):
    with open(filename, 'r') as f:
        return {index: line.strip() for index, line in enumerate(f.readlines())}

labels = load_labels(label_path)

# Initialize TFLite interpreter and allocate tensors
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {input_path}")

# Prepare output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to preprocess the input image
def preprocess_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    return input_data.astype(np.uint8)  # Corrected to UINT8

# Function to draw bounding boxes and labels on the frame
def draw_detections(frame, results):
    for obj in results:
        if obj['score'] > confidence_threshold:
            box = obj['bounding_box']
            label = f"{labels[obj['class_id']]}: {obj['score']:.2f}"
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    results = [{'bounding_box': box, 'class_id': int(class_id), 'score': score} for box, class_id, score in zip(boxes, class_ids, scores)]
    
    draw_detections(frame, results)
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()