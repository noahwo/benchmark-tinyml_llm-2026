import cv2
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize interpreter
interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]), Image.ANTIALIAS)
    input_data = np.array(img).reshape(input_details[0]['shape'])

    # Set tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))

    # Run inference
    interpreter.invoke()

    # Extract output data from the interpreter
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw bounding boxes and labels on the frame
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * cap.get(3), xmax * cap.get(3),
                                          ymin * cap.get(4), ymax * cap.get(4))
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()