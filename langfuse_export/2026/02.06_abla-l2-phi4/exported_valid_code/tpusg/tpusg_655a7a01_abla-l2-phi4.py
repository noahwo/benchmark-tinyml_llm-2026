import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(label_path)

# Setup TFLite interpreter
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing function
def preprocess_frame(frame):
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
    return input_data

# Inference function
def run_inference(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    
    return boxes, classes, scores

# Output handling and drawing function
def draw_detections(frame, boxes, classes, scores):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            
            # Draw rectangle and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Open input video
cap = cv2.VideoCapture(input_path)
output_video = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Run inference
    boxes, classes, scores = run_inference(input_data)

    # Draw detections on the frame
    draw_detections(frame, boxes, classes, scores)
    
    # Initialize video writer if not already done
    if output_video is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width, _ = frame.shape
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write the processed frame to the output video
    output_video.write(frame)

# Release resources
cap.release()
output_video.release()

print("Processing complete. Output saved to:", output_path)