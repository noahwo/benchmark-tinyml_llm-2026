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

# Load labels
labels = read_label_file(label_path)

# Initialize the interpreter with Edge TPU support
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine the required image size for the model
_, height, width, _ = input_details[0]['shape']

# Open video file for reading
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to set up the output video writer
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB and resize it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_rgb_frame = Image.fromarray(cv2.resize(rgb_frame, (width, height)))

    # Prepare input tensor
    input_tensor = np.array(resized_rgb_frame).reshape((1, height, width, 3))
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run the inference
    interpreter.invoke()

    # Get detection results
    objs = get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes and labels on the original frame
    for obj in objs:
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)
        x1, y1 = int(bbox.xmax * frame_width), int(bbox.ymax * frame_height)

        # Draw bounding box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Get label and score
        label = labels.get(obj.id, obj.id)
        score = int(100 * obj.score)

        # Draw label text
        label_text = f'{label}: {score}%'
        cv2.putText(frame, label_text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detections to output video
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()