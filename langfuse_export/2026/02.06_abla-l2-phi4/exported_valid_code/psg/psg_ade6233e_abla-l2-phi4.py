import os
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']

# Video capture and output setup
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def preprocess_frame(frame):
    # Resize the frame to match model input size
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    
    # Normalize and convert to UINT8
    frame_normalized = frame_resized / 255.0
    frame_uint8 = (frame_normalized * 255).astype(np.uint8)
    
    return np.expand_dims(frame_uint8, axis=0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input frame
    input_data = preprocess_frame(frame)

    # Set model input tensor
    interpreter.set_tensor(input_index, input_data)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    elapsed_ms = (time.time() - start_time) * 1000

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    frame_height, frame_width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            
            # Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)
    print(f"Frame processed in {elapsed_ms:.2f} ms")

cap.release()
out.release()

print("Processing complete.")