import cv2
import numpy as np

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter

def load_labels(label_file):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(label_path)
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def preprocess_frame(frame):
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(input_details[0]['dtype'])
    return input_data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = preprocess_frame(frame)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
            cv2.putText(frame, label, (int(left), int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()