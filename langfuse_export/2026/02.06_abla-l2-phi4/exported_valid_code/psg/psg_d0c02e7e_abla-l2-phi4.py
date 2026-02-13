import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from label map
def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Initialize TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# Preprocessing function
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    return input_data

# Inference function
def detect_objects(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects
    
    return boxes, classes, scores

# Video processing
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    input_data = preprocess_frame(frame)
    
    start_time = time.time()
    boxes, classes, scores = detect_objects(input_data)
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Inference time: {elapsed_ms:.2f} ms")
    
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * cap.get(3)), int(xmax * cap.get(3)), int(ymin * cap.get(4)), int(ymax * cap.get(4))
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
            cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()

print("Video processing complete.")