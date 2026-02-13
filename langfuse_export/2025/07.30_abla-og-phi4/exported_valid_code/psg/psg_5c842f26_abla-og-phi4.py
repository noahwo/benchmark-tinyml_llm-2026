import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Input Data
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0).astype(input_dtype)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Phase 4: Output Interpretation & Handling Loop
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = int(classes[i])
            box = boxes[i]
            
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)

            label = f"{labels[class_id]}: {scores[i]:.2f}"
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()