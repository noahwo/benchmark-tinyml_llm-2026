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
with open(label_path, 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Initialize TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video from the given path
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError("Cannot open video file")

# Prepare video writer for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Preprocessing and inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize the image to required input shape
    input_shape = input_details[0]['shape']
    height, width = input_shape[1:3]
    img_resized = cv2.resize(frame, (width, height))
    img_normalized = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Set tensor to the model and run inference
    interpreter.set_tensor(input_details[0]['index'], img_normalized)
    interpreter.invoke()

    # Extract results from output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int64)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw detection boxes and labels on the frame
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            label = labels[classes[i]]
            score = scores[i]
            
            # Convert normalized coordinates to pixel values
            (xminn, xmaxx, yminn, ymaxx) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(frame, (int(xminn), int(yminn)), (int(xmaxx), int(ymaxx)), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (int(xminn), int(yminn) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detection to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()