"""
Extracted from: raw_qwen2.5-coder:32b_1974_psg_batch
Entry ID: efd7e873
Entry Name: efd7_py_sketch_generator
Session ID: qwen2.5-coder:32b_1974_psg_batch
Timestamp: 2025-07-30T09:17:55.474000+00:00
Tags: benchmark, py_sketch_generator, qwen2.5-coder:32b
"""

# Phase 1: Setup

# 1.1 Imports
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections_efd7.mp4"

# 1.3 Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop

# 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2.2 Preprocess Data
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    input_data = input_data.astype(input_details[0]['dtype'])

    # Phase 3: Inference

    # 3.1 Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 3.2 Run Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # 4.1 Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 4.2 Interpret Results
    detections = np.squeeze(output_data)

    for detection in detections:
        if detection[2] > 0.5:  # Assuming the third element is confidence score and thresholding at 0.5
            label_index = int(detection[1])
            label = labels[label_index]
            box = detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # 4.3 Handle Output
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detection to the output video file
    out.write(frame)

# Phase 5: Cleanup

cap.release()
out.release()
cv2.destroyAllWindows()