import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import numpy as np

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter with Coral Edge TPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input image size
_, input_height, input_width, _ = input_details[0]['shape']

def set_input_tensor(image):
    """Sets the input tensor."""
    image = cv2.resize(image, (input_width, input_height))
    image = np.expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], image)

def get_output():
    """Returns the detection results."""
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    return boxes, classes, scores

# Open video input and output streams
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    set_input_tensor(frame)

    interpreter.invoke()

    boxes, classes, scores = get_output()

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            # Get bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()