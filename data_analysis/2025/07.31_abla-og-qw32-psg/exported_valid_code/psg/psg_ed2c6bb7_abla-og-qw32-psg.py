import numpy as np
import time
import cv2

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
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
    num_detections = interpreter.get_tensor(output_details[3]['index'])[0]

    for i in range(int(num_detections)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]

            ymin, xmin, ymax, xmax = box
            (xminn, xmaxx, yminn, ymaxx) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(frame, (int(xminn), int(yminn)), (int(xmaxx), int(ymaxx)), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (int(xminn), int(yminn) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()