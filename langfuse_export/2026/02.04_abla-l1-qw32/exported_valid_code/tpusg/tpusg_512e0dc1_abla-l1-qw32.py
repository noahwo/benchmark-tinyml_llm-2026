import cv2
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
import numpy as np

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

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize and normalize image
    resized_img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]), Image.ANTIALIAS)
    
    # Set the tensor to point to the new frame data
    common.set_input(interpreter, resized_img)

    # Run inference
    interpreter.invoke()

    # Extract results
    objs = detect.get_objects(interpreter, score_threshold=confidence_threshold)

    # Draw bounding boxes and labels on the original frame
    for obj in objs:
        bbox = obj.bbox
        label = labels.get(obj.id, 'Unknown')
        cv2.rectangle(frame, (int(bbox.xmin * img.width), int(bbox.ymin * img.height)),
                      (int(bbox.xmax * img.width), int(bbox.ymax * img.height)), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {obj.score:.2f}', (int(bbox.xmin * img.width), int(bbox.ymin * img.height) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()