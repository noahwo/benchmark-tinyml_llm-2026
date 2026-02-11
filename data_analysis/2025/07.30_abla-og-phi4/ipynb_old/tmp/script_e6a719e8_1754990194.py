"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: cf6ff531
Entry Name: cf6f_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T14:26:18.995000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_cf6f.mp4"
confidence_threshold  = 0.5

# Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e:
    print(f"Failed to load Edge TPU delegate: {e}. Trying alternative path...")
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Edge TPU delegate: {e}")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get model input size
expected_height, expected_width = input_details[0]['shape'][1], input_details[0]['shape'][2]

# Video Capture
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Could not open video file")

# Video Writer
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input size
    resized_frame = cv2.resize(frame, (expected_width, expected_height))
    
    # Preprocess Frame
    input_tensor = np.expand_dims(resized_frame, axis=0)

    # Set Input Tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run Inference
    interpreter.invoke()

    # Get Output Tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence score

    # Interpret Results and Draw Boxes on original frame size
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (
                int(xmin * width),
                int(xmax * width),
                int(ymin * height),
                int(ymax * height)
            )
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write the frame with detection boxes
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()