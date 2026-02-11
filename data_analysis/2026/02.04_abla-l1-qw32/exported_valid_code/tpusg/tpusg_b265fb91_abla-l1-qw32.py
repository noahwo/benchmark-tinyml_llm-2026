import cv2
import numpy as np
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Initialize the interpreter
interpreter = make_interpreter(model_path, device=':0')
interpreter.allocate_tensors()

# Read labels
labels = read_label_file(label_path)

# Open input video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image and preprocess it
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
    common.set_input(interpreter, resized_img)

    # Run inference
    interpreter.invoke()

    # Get detection results
    detections = detect.get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes on the frame
    for obj in detections:
        bbox = obj.bbox
        x0, y0, x1, y1 = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height), int(bbox.xmax * frame_width), int(bbox.ymax * frame_height)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f'{labels.get(obj.id, obj.id)}: {obj.score:.2f}'
        cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    out.write(frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()