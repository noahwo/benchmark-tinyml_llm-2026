import os
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from label map file
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

# Initialize TFLite interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input details and output details from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# Load labels
labels = load_labels(label_path)

# Open video capture and writer
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Function to preprocess the input image
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    return input_data

# Inference and postprocessing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Draw detection boxes and labels on the frame
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i]*100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*label_size[1]))),
                          (int(left + round(1.5*label_size[0])), int(top + base_line)), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top + base_line)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with detections to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved at:", output_path)