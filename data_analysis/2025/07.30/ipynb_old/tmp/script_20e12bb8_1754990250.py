"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: a7d40f6a
Entry Name: a7d4_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T14:24:26.509000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_a7d4.mp4"
confidence_threshold   = 0.5

# Phase 1: Setup
try:
    # Load labels
    with open(label_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]

    # Initialize interpreter with EdgeTPU delegate
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load Edge TPU delegate: {e}. Trying alternative path.")
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])

    # Allocate tensors
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

except Exception as e:
    print(f"Error during setup: {e}")
    exit(1)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (int(cap.get(3)), int(cap.get(4))))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the input image to match the model's expected size
    input_size = input_details[0]['shape'][2]
    resized_frame = cv2.resize(frame, (input_size, input_size))

    # Preprocess input data
    image_np = np.expand_dims(resized_frame, axis=0)
    
    # Quantization handling
    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        image_np = (np.float32(image_np) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Interpret results and draw bounding boxes
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            
            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]
            label = f"{object_name}: {int(scores[i]*100)}%"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*label_size[1]))),
                          (int(left + round(1.5*label_size[0])), int(top + base_line)),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with bounding boxes to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()