import numpy as np
import time
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Setup
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter with EdgeTPU acceleration
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
import cv2

cap = cv2.VideoCapture(input_path)
video_writer = None
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))
    
    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            label = labels[int(classes[i])]
            
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', p1, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Handle output
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 
                                       cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
    
    video_writer.write(frame)
    frame_count += 1

# Phase 5: Cleanup
cap.release()
if video_writer is not None:
    video_writer.release()

print(f'Processed {frame_count} frames and saved to {output_path}')