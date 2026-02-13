import cv2
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
from pycoral.adapters import common, detect

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter with Edge TPU
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load video
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

    # Preprocess the frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.array(resized_img)
    common.set_input(interpreter, input_data)

    # Run inference
    interpreter.invoke()

    # Get detection results
    detections = detect.get_objects(interpreter, confidence_threshold)

    # Draw bounding boxes and labels on the frame
    for obj in detections:
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * width), int(bbox.ymin * height)
        x1, y1 = int(bbox.xmax * width), int(bbox.ymax * height)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f'{labels[obj.id]}: {obj.score:.2f}'
        cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()