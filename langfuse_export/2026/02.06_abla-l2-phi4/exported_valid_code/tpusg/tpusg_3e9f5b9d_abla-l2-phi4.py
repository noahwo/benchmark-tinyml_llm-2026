import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def load_tflite_model(model_file):
    interpreter = Interpreter(
        model_path=model_file,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()
    return interpreter

# Setup
interpreter = load_tflite_model(model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing function
def preprocess_frame(frame):
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    return input_data

# Inference function
def infer(interpreter, frame):
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects
    
    return boxes, classes, scores

# Output handling function
def draw_boxes(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i]*100)}%"
            cv2.putText(frame, label, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Read input video
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
    
    boxes, classes, scores = infer(interpreter, frame)
    draw_boxes(frame, boxes, classes, scores)

    out.write(frame)  # Save the output frame

cap.release()
out.release()

# Note: mAP calculation is a complex task and requires more context like ground truth data. It's not implemented here.