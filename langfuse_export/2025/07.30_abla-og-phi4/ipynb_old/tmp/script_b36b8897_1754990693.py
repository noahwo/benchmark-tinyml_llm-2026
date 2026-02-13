"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: dd530656
Entry Name: dd53_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T14:13:30.032000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import cv2

# Configuration parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_dd53.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Load labels
def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(label_path)

# Load TFLite model and allocate tensors.
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except ValueError:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )

interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
def preprocess_frame(frame, input_shape):
    image_height, image_width, _ = frame.shape
    input_image_size = input_details[0]['shape'][1]
    
    frame = cv2.resize(frame, (input_image_size, input_image_size))
    frame = np.expand_dims(frame, axis=0)
    
    if input_details[0]['dtype'] == np.float32:
        frame = (np.float32(frame) - 127.5) / 127.5
    
    return frame

# Open video file or capture device
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (input_details[0]['shape'][2], input_details[0]['shape'][1]))

# Phase 3 and 4: Inference & Output Interpretation & Handling Loop
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame
    input_data = preprocess_frame(frame, input_details[0]['shape'][1:])
    
    # Set model input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    
    h, w, _ = frame.shape
    
    # Draw the results on the frame
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
            
            # Draw a bounding box.
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            
            # Draw label
            label = f'{labels[int(classes[i])]}: {int(scores[i] * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*label_size[1]))), 
                          (int(left + label_size[0]), int(top + base_line)), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top + base_line)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Write the frame with detections
    out.write(frame)
    
    # Display the frame for debugging purposes
    cv2.imshow('Object detector', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()