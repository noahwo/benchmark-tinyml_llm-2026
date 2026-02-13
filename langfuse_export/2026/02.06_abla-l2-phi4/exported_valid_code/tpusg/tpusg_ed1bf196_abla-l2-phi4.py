import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import os

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from label map file
def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

# Initialize the TFLite interpreter
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read video from the given path
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

labels = load_labels(label_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Process output
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {int(scores[i] * 100)}%'
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Note: mAP calculation is not implemented here as it requires multiple frames and ground truth data.