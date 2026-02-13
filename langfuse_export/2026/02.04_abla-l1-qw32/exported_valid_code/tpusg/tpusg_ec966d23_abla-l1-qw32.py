import cv2
import numpy as np
from PIL import Image
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.detect import get_objects
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
labels = read_label_file(label_path)
inference_size = input_size(interpreter)

# Open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit('ERROR: Unable to open video source.')

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def draw_objects(frame, objs, labels):
    for obj in objs:
        bbox = obj.bbox
        label = f'{labels.get(obj.id, obj.id)}: {obj.score:.2f}'
        if obj.score >= confidence_threshold:
            cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox.xmin, bbox.ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_img = Image.fromarray(cv2.resize(rgb_frame, inference_size))

    # Run inference
    set_input(interpreter, resized_img)
    interpreter.invoke()
    objs = get_objects(interpreter, confidence_threshold)

    # Draw the results on the frame
    draw_objects(frame, objs, labels)

    # Write the frame to output video file
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()