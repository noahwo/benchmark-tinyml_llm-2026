import numpy as np
import cv2

# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter

# Paths/Parameters
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

# Video Writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_shape = input_details[0]['shape']
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized.astype(input_details[0]['dtype'])

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = interpreter.get_tensor(output_details[3]['index'])

    # Interpret Results
    for i in range(int(num_detections)):
        if scores[0][i] > confidence_threshold:
            box = boxes[0][i]
            class_id = int(classes[0][i])
            label = labels[class_id]

            # Handle Output: Draw bounding box and label
            height, width, _ = frame.shape
            ymin = int(max(1, (box[0] * height)))
            xmin = int(max(1, (box[1] * width)))
            ymax = int(min(height, (box[2] * height)))
            xmax = int(min(width, (box[3] * width)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {int(scores[0][i]*100)}%', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()