import cv2
import numpy as np

# CONFIGURATION PARAMETERS
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to match model input size
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    
    # Normalize the image to 0-1 range
    input_data = np.expand_dims(resized_frame / 255.0, axis=0).astype(input_dtype)
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        
    out.write(frame)

# Cleanup
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()