import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Setup and load TFLite interpreter
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels from label map file
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_frame(frame):
    # Resize frame to model's expected input size and convert it to UINT8 type
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
    return input_data

def draw_detections(frame, detections):
    for detection in detections:
        ymin, xmin, ymax, xmax, score, class_id = detection
        if score >= confidence_threshold:
            label = f"{labels[int(class_id)]}: {int(score * 100)}%"
            start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
            cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
            cv2.putText(frame, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Open input and output video files
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Collect detections
    detections = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            detections.append((ymin, xmin, ymax, xmax, scores[i], classes[i]))

    # Draw detections on the frame
    draw_detections(frame, detections)

    # Write the processed frame to output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved at:", output_path)