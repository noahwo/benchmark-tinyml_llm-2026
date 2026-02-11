# Phase 1: Setup

## 1.1 Imports
import numpy as np
from tflite_runtime.interpreter import Interpreter
import cv2

## 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

## 1.3 Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

## 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

## 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop

## 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError(f"Could not open video file {input_path}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

## Loop Control
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    ## 2.2 Preprocess Data
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    input_data = input_data.astype(input_details[0]['dtype'])

    # Phase 3: Inference

    ## 3.1 Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    ## 3.2 Run Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    ## 4.1 Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    ## 4.2 Interpret Results
    # Assuming the model outputs a classification result, we'll interpret it here.
    # This is a placeholder for actual interpretation logic based on your model's output format.
    predicted_index = np.argmax(output_data)
    predicted_label = labels[predicted_index]
    
    ## 4.3 Handle Output
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(frame)

# Phase 5: Cleanup

## Release resources
cap.release()
out.release()
cv2.destroyAllWindows()