import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# Phase 1: Setup
## 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

## 1.3 Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

## 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

## 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
## 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError("Error opening video file")

## Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

# Phase 3: Inference and Phase 4: Output Interpretation & Handling Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ## 2.2 Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    ## 3.1 Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    ## 3.2 Run Inference
    interpreter.invoke()

    ## 4.1 Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    ## 4.2 Interpret Results
    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * cap.get(3), xmax * cap.get(3),
                                          ymin * cap.get(4), ymax * cap.get(4))
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]

            ## Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    ## 4.3 Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()