import os
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import cv2

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Setup Interpreter with EdgeTPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read Video
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    input_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(input_frame, axis=0).astype(np.uint8)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Output handling
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Put label
            object_name = labels[int(classes[i])] if classes[i] >= 0 else 'N/A'
            label = f'{object_name}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)
    print(f'Frame processed in {inference_time:.4f} seconds')

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Note: mAP calculation is not included here as it requires ground truth data and additional implementation.