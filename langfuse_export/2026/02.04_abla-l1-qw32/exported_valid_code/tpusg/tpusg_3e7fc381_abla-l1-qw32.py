import cv2
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
def load_labels(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(label_path)

# Initialize interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Open input video file
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError("Cannot open video file")

# Get video properties
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

    # Convert frame to RGB and resize for model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = Image.fromarray(rgb_frame).resize((300, 300), Image.ANTIALIAS)

    # Set tensor to the interpreter and run inference
    common.set_input(interpreter, input_tensor)
    interpreter.invoke()

    # Get detection results
    detections = detect.get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes and labels on frame
    for obj in detections:
        bbox = obj.bbox
        score = obj.score
        label = labels[obj.id]

        # Convert normalized coordinates to image coordinates
        ymin, xmin, ymax, xmax = bbox
        ymin, ymax = int(ymin * frame_height), int(ymax * frame_height)
        xmin, xmax = int(xmin * frame_width), int(xmax * frame_width)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label_text = f'{label}: {score:.2f}'
        cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()