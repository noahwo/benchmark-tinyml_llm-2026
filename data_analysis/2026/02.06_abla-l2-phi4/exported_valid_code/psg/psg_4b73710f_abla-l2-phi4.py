import os
import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_shape[1], input_shape[2]))
    return np.expand_dims(frame_resized, axis=0)

def postprocess_output(output_data, image_height, image_width):
    boxes = output_data[0][0]
    classes = output_data[1][0].astype(np.int32)
    scores = output_data[2][0]

    results = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image_width, xmax * image_width,
                                          ymin * image_height, ymax * image_height)

            results.append((int(left), int(top), int(right), int(bottom),
                            scores[i], classes[i]))

    return results

def draw_boxes(frame, boxes):
    for box in boxes:
        left, top, right, bottom, score, class_id = box
        label = f"{labels[class_id]}: {score:.2f}"
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

# Video processing
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError("Error opening video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                      (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000

    output_data = [interpreter.get_tensor(output['index']) for output in output_details]
    detections = postprocess_output(output_data, frame_height, frame_width)
    
    draw_boxes(frame, detections)

    out.write(frame)
    print(f"Inference Time: {inference_time:.2f} ms")

cap.release()
out.release()