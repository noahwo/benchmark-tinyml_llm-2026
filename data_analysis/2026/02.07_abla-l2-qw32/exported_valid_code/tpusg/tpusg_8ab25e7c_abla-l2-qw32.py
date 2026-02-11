import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# CONFIGURATION PARAMETERS
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Setup: Load the TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

def preprocess_image(image):
    # Resize image to input size
    input_shape = input_details[0]['shape']
    resize_img = cv2.resize(image, (input_shape[1], input_shape[2]))
    # Normalize pixel values if necessary
    resize_img = np.expand_dims(resize_img, axis=0)
    return resize_img

def postprocess_detection(results, frame):
    boxes = results['detection_boxes'][0]
    classes = results['detection_classes'][0].astype(int) + 1
    scores = results['detection_scores'][0]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            # Get bounding box coordinates and label
            ymin, xmin, ymax, xmax = boxes[i]
            label = labels[classes[i]]
            score = scores[i]

            # Draw rectangle and text on frame
            h, w, _ = frame.shape
            (left, right, top, bottom) = (xmin * w, xmax * w,
                                          ymin * h, ymax * h)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),
                          (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input data
    input_data = preprocess_image(frame)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Post-process the output
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = interpreter.get_tensor(output_details[3]['index'])

    results = {
        'detection_boxes': boxes,
        'detection_classes': classes,
        'detection_scores': scores,
        'num_detections': num_detections
    }

    postprocess_detection(results, frame)

    # Write the processed frame to output video file
    out.write(frame)
    
    # Print inference time (optional)
    print(f'Inference Time: {end_time - start_time:.3f} seconds')

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()