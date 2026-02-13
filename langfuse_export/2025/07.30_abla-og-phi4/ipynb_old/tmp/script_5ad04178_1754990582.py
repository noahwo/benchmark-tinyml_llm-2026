"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: bd5ca91c
Entry Name: bd5c_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T14:16:07.100000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import cv2

# Configuration Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_bd5c.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e:
    print(f"Failed to load Edge TPU delegate: {e}")
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        raise RuntimeError("Edge TPU library not found. Ensure it's installed and accessible.") from e

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input size
_, input_height, input_width, _ = input_details[0]['shape']

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Frame
    image_data = cv2.resize(frame, (input_width, input_height))
    image_data = np.expand_dims(image_data, axis=0)

    # Quantization Handling
    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        image_data = (np.float32(image_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*label_size[1]))),
                          (int(left + round(1.5*label_size[0])), int(top + base_line)),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top + base_line)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()