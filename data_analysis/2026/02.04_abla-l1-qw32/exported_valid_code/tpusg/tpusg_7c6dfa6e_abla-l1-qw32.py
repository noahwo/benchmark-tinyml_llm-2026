import os
import cv2
import numpy as np
from PIL import Image
from pycoral.adapters.common import input_size
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
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load labels
labels = read_label_file(label_path)

# Get input size
width, height = input_size(interpreter)

# Open video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit('ERROR: Unable to open video source.')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and prepare the image
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((width, height), Image.ANTIALIAS)
    scale = (frame.shape[1] / width, frame.shape[0] / height)

    # Run inference
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.array(im).reshape(1, width, height, 3))
    interpreter.invoke()

    # Get detected objects
    objs = get_objects(interpreter, confidence_threshold)
    
    # Draw results on the frame
    for obj in objs:
        bbox = obj.bbox
        bbox_scaled = (int(bbox.xmin * scale[0]), int(bbox.ymin * scale[1]),
                       int(bbox.width * scale[0]), int(bbox.height * scale[1]))
        
        cv2.rectangle(frame, (bbox_scaled[0], bbox_scaled[1]), 
                      (bbox_scaled[0] + bbox_scaled[2], bbox_scaled[1] + bbox_scaled[3]), 
                      (0, 255, 0), 2)
        
        label = f'{labels.get(obj.id, obj.id)}: {obj.score:.2f}'
        cv2.putText(frame, label, (bbox_scaled[0], bbox_scaled[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()