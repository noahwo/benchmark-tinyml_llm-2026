import os
import cv2
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
def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(label_path)

# Initialize the interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Open video file
cap = cv2.VideoCapture(input_path)
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

    # Convert the frame to RGB and resize it for the model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    resized_img = img.resize((common.input_size(interpreter)[0], common.input_size(interpreter)[1]), Image.ANTIALIAS)

    # Run inference
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    detections = detect.get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes and labels on the original frame
    for obj in detections:
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * width), int(bbox.ymin * height)
        x1, y1 = int(bbox.xmax * width), int(bbox.ymax * height)
        label = f'{labels.get(obj.id, "unknown")}: {obj.score:.2f}'
        
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()