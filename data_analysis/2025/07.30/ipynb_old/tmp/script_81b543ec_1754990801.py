"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: 95834f79
Entry Name: 9583_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T14:09:52.931000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import os
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Import cv2 globally
import cv2

# Configuration Parameters
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections_9583.mp4"
confidence_threshold   = 0.5

# Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

def load_labels(path):
    with open(path, 'r') as f:
        return [l.strip() for l in f.readlines()]

def load_interpreter():
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
    except Exception as e:
        print(f"Failed to load EdgeTPU delegate: {e}")
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )

    interpreter.allocate_tensors()
    return interpreter

def preprocess_input(interpreter, input_data):
    input_details = interpreter.get_input_details()
    floating_model = (input_details[0]['dtype'] == np.float32)
    
    # Assuming the input_data is a video frame or image
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(input_data, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data

def postprocess_output(interpreter, frame):
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            label = labels[int(classes[i])]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            text = f'{label}: {int(scores[i] * 100)}%'
            cv2.putText(frame, text, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    interpreter = load_interpreter()
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_input(interpreter, frame)
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()

        postprocess_output(interpreter, frame)
        
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()