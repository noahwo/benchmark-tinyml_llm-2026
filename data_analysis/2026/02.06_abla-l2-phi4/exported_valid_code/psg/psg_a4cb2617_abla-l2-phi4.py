import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load Labels
def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Setup TFLite Interpreter
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']
output_indices = [detail['index'] for detail in output_details]

# Video Processing
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Inference
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Extract output data
    boxes = interpreter.get_tensor(output_indices[0])
    classes = interpreter.get_tensor(output_indices[1])
    scores = interpreter.get_tensor(output_indices[2])

    height, width, _ = frame.shape

    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            class_id = classes[0][i]

            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            label_text = f"{labels[int(class_id)]}: {scores[0][i]:.2f}"
            cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Output handling
    out.write(frame)

cap.release()
out.release()