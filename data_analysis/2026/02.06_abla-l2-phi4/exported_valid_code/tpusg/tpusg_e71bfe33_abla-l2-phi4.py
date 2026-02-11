import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import os

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(path):
    with open(path, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Setup TFLite interpreter
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing function for video frames
def preprocess_frame(frame, input_shape):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_shape[1], input_shape[0]))
    return np.expand_dims(frame_resized, axis=0)

# Postprocessing function to draw detection boxes
def draw_boxes(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            left, right, top, bottom = int(xmin * frame.shape[1]), int(xmax * frame.shape[1]), int(ymin * frame.shape[0]), int(ymax * frame.shape[0])

            cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
            cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Open video files
input_video = cv2.VideoCapture(input_path)
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                               input_video.get(cv2.CAP_PROP_FPS),
                               (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Process each frame
while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame, interpreter.get_input_details()[0]['shape'][1:3])
    
    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    interpreter.invoke()

    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Draw detection boxes on the frame
    draw_boxes(frame, boxes, classes, scores)

    # Write the frame to output video
    output_video.write(frame)

# Release resources
input_video.release()
output_video.release()

print("Video processing completed and saved at:", output_path)