import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open input video")

# Prepare to write the output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def preprocess_image(frame):
    # Ensure the frame is in BGR format and convert it to RGB
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("Invalid image format")
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data.astype(np.uint8)

def draw_detections(frame, boxes, classes, scores):
    for box, cls, score in zip(boxes, classes, scores):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = box
            h, w, _ = frame.shape
            (left, right, top, bottom) = (int(xmin * w), int(xmax * w),
                                          int(ymin * h), int(ymax * h))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{labels[int(cls)]}: {score:.2f}"
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    
    # Set tensor to point to the preprocessed image data.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    draw_detections(frame, boxes[0], classes[0], scores[0])
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()