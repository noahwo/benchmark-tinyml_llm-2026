"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: 392aed53
Entry Name: 392a_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T13:46:58.109000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# CONFIGURATION PARAMETERS
model_path    = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path    = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path    = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_392a.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
def load_labels(label_file):
    with open(label_file, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except ValueError:
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        print(f"Failed to load Edge TPU delegate: {e}")
        exit(1)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check expected input type and shape
expected_input_type = input_details[0]['dtype']
print(f"Expected input type: {expected_input_type}")

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (300, 300))
    
    # Convert to UINT8 if expected by model
    if expected_input_type == np.uint8:
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
    else:
        raise ValueError("Unexpected input type for the model.")
    
    return input_data

# Phase 4: Output Interpretation & Handling Loop
def interpret_output(output_data):
    boxes = output_data[0][0]
    classes = output_data[1][0]
    scores = output_data[2][0]

    results = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            # Adjust tuple to include only box, score, and class_id
            results.append(((int(left), int(top), int(right), int(bottom)), scores[i], classes[i]))
    return results

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input
    input_data = preprocess_frame(frame)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4.1: Get Output Tensor(s)
    output_data = [
        interpreter.get_tensor(output_details[i]['index'])
        for i in range(len(output_details))
    ]

    # Phase 4.2 & 4.3: Interpret Results
    results = interpret_output(output_data)

    # Draw bounding boxes and labels on the frame
    for (box, score, class_id) in results:
        left, top, right, bottom = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{labels[int(class_id)]}: {int(score * 100)}%"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()