import cv2
import numpy as np
from PIL import Image
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
labels = read_label_file(label_path)

# Initialize interpreter and allocate tensors
interpreter = make_interpreter(model_path, device=':0')  # Use Edge TPU
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# Open video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit('Cannot open video')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image and resize it
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((input_width, input_height), Image.ANTIALIAS)

    # Run inference
    common.set_input(interpreter, img)
    interpreter.invoke()
    objs = detect.get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes and labels on the frame
    for obj in objs:
        bbox = obj.bbox
        label = f'{labels.get(obj.id, obj.id)}: {obj.score:.2f}'
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
        cv2.putText(frame, label, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()