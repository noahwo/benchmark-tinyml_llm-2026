import numpy as np
import os
from tflite_runtime.interpreter import Interpreter, load_delegate
import cv2

# Phase 1: Setup
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load libedgetpu.so.1.0: {e}")
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0: {e}")
        interpreter = Interpreter(model_path=model_path)

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height, width = input_details[0]['shape'][1:3]
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    for detection in output_data[0]:
        score = float(detection[2])
        if score > confidence_threshold:
            bbox = detection[3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (startX, startY, endX, endY) = bbox.astype("int")
            
            label = labels[int(detection[1])]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}".format(label, score)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()