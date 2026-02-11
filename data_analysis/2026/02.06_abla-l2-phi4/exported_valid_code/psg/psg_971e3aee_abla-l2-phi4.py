import os
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing function
def preprocess_frame(frame):
    input_shape = input_details[0]['shape']
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
    return input_data

# Inference function
def run_inference(input_frame):
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()
    
    # Get output details
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    return boxes, classes, scores

# Post-processing and drawing function
def draw_detections(frame, boxes, classes, scores):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = int(classes[i])
            box = tuple(boxes[i].tolist())
            
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = (xmin * width, xmax * width,
                                        ymin * height, ymax * height)
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[class_id]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Video processing
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_frame(frame)
    boxes, classes, scores = run_inference(input_data)
    draw_detections(frame, boxes, classes, scores)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# Note: Calculation of mAP is complex and requires additional implementation
# based on the specific requirements and dataset annotations.