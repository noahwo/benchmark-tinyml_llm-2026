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
    raise SystemExit('ERROR: Video failed to open.')

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]), Image.ANTIALIAS)
    input_data = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw bounding boxes and labels on the frame
    for i, score in enumerate(scores):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            label = '%s: %d%%' % (labels[int(classes[i])], int(score * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, labelSize[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*labelSize[1]))),
                          (int(left + round(1.5*labelSize[0])), int(top + baseLine)),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

    # Write the frame to the output file
    out.write(frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()