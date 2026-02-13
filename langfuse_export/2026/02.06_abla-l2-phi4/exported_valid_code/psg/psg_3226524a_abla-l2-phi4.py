import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']

# Video processing
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(image_resized, axis=0)

    # Set the tensor to the interpreter
    input_tensor_type = input_details[0]['dtype']
    if input_tensor_type == np.uint8:
        input_data = input_data.astype(np.uint8)
    elif input_tensor_type == np.float32:
        input_data = (input_data / 255.0).astype(np.float32)

    interpreter.set_tensor(input_index, input_data)

    # Inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Output handling
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)

            # Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {scores[i]:.2f}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - label_size[1])), 
                          (int(left + label_size[0]), int(top + base_line)), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with detection boxes
    out.write(frame)

# Release resources
cap.release()
out.release()

print("Object detection complete.")