import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']

# Preprocessing function for the video frames
def preprocess_frame(frame, target_shape):
    frame_resized = cv2.resize(frame, (target_shape[1], target_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
    return input_data

# Read and process video
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame, input_shape)

    # Run inference
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])

            # Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {scores[i]:.2f}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top) - round(1.5 * label_size[1])), 
                          (int(left) + round(1.5 * label_size[0]), int(top + base_line)), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top) + base_line), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with bounding boxes
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Note: Calculating mAP requires a more complex setup and is not included in this script.