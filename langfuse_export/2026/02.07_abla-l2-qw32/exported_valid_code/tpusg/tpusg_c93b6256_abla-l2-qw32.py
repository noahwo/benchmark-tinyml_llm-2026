import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(filename):
    with open(filename, 'r', encoding='utf8') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(label_path)

# Initialize interpreter with EdgeTPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise Exception("Error opening video stream or file")

# Get video properties to save processed video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess_image(image):
    # Resize to input size required by model
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(image, (input_shape[1], input_shape[2]))
    
    # Convert to uint8 if needed and add batch dimension
    input_data = np.expand_dims(image_resized.astype(np.uint8), axis=0)
    
    return input_data

def postprocess_output(output_details):
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))
    
    return boxes, classes, scores, num_detections

# Main inference loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess image
    input_data = preprocess_image(frame)
    
    # Set tensor and invoke interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Post-process output
    boxes, classes, scores, num_detections = postprocess_output(output_details)
    
    # Draw detections on frame
    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            score = scores[i]
            
            label = labels.get(class_id, 'Unknown')
            display_text = f'{label}: {int(score * 100)}%'
            
            ymin, xmin, ymax, xmax = box
            im_height, im_width, _ = frame.shape
            
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Write the frame into the file 'output_path'
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processed video saved to", output_path)