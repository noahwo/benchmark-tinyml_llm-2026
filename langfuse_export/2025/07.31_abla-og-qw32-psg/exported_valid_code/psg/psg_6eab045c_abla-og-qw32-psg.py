import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            box = boxes[0][i]
            class_id = int(classes[0][i])
            label = labels[class_id]

            # Calculate bounding box coordinates
            y_min, x_min, y_max, x_max = box
            im_height, im_width, _ = frame.shape
            (left, right, top, bottom) = (x_min * im_width, x_max * im_width,
                                          y_min * im_height, y_max * im_height)

            # Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[0][i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()