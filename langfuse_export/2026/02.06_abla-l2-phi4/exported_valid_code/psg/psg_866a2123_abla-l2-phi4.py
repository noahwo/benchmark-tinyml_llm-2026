import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

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

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Video processing
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    return input_data

def postprocess_output(interpreter, frame):
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocessing
    input_data = preprocess_frame(frame)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Post-processing and drawing results on frame
    output_frame = postprocess_output(interpreter, frame)
    
    # Write the frame with detection boxes to output video
    out.write(output_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed. Output saved at:", output_path)