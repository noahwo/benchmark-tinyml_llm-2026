import cv2
from PIL import Image
import numpy as np
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

# Read labels
labels = read_label_file(label_path)

# Open input video
cap = cv2.VideoCapture(input_path)

# Get video properties to save output video with the same properties
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

    # Convert the frame to a PIL image and resize it
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize(input_size(interpreter), Image.ANTIALIAS)

    # Run inference
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.array(resized_img).reshape(1, *resized_img.size, 3))
    interpreter.invoke()

    # Get detection results
    objs = get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes and labels on the frame
    for obj in objs:
        bbox = obj.bbox
        score = obj.score
        label = labels.get(obj.id, 'Unknown')

        # Scale bounding box coordinates to original image dimensions
        xmin = int(bbox.xmin * frame_width)
        xmax = int(bbox.xmax * frame_width)
        ymin = int(bbox.ymin * frame_height)
        ymax = int(bbox.ymax * frame_height)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label_text = f'{label}: {score:.2f}'
        cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()