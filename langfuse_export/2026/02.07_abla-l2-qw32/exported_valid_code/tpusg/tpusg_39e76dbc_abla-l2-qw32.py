import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(filename):
    with open(filename, 'r') as f:
        return {index: line.strip() for index, line in enumerate(f)}

labels = load_labels(label_path)

# Initialize TFLite interpreter with EdgeTPU
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the image
def preprocess_image(frame):
    frame_resized = cv2.resize(frame, (300, 300))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)  # Ensure the data type is uint8
    return input_data

# Read video from the given path
cap = cv2.VideoCapture(input_path)

# Prepare to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Inference and output handling
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_image(frame)
    
    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    # Draw detection boxes and labels
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label_text = f"{object_name}: {scores[i]:.2f}"
            cv2.putText(frame, label_text, (int(left), int(top) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video with detections has been saved to:", output_path)