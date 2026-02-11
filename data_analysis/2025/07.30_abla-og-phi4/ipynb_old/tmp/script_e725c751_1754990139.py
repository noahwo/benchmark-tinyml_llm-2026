"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: cd1e42b6
Entry Name: cd1e_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T14:28:04.053000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_cd1e.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Load labels
def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(label_path)

# Load interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load delegate 'libedgetpu.so.1.0': {e}")
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        raise RuntimeError(f"Failed to load Edge TPU delegate: {e}")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (300, 300))
    input_data = np.expand_dims(img_resized, axis=0)

    # Quantization Handling
    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        input_scale, input_zero_point = input_details[0]["quantization"]
        input_data = input_data / input_scale + input_zero_point

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])] if classes[i] >= 0 else 'N/A'
            label = f'{object_name}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write the frame with bounding boxes to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Phase 5: Cleanup