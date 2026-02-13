import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize video capture and writer
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

def preprocess(frame):
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    return input_data

def postprocess(frame, interpreter_results):
    boxes = interpreter_results[0][0]  # Bounding box coordinates of detected objects
    classes = interpreter_results[1][0]  # Class index of detected objects
    scores = interpreter_results[2][0]  # Confidence of detected objects
    
    h, w, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess(frame)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    interpreter_results = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    postprocess(frame, interpreter_results)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()