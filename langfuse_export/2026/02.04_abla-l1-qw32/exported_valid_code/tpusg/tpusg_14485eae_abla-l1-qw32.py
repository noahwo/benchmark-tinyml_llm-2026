import cv2
from PIL import Image
import numpy as np
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Initialize the interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load labels
labels = read_label_file(label_path)

# Open input video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit('ERROR: Unable to open input video')

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image and resize to model input size
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)

    # Run inference
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    objs = detect.get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes on the frame
    for obj in objs:
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * width), int(bbox.ymin * height)
        x1, y1 = int(bbox.xmax * width), int(bbox.ymax * height)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Add label and confidence
        label = labels.get(obj.id, obj.id)
        text = f'{label}: {obj.score:.2f}'
        cv2.putText(frame, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()