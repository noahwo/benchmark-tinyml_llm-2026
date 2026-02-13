import cv2
import numpy as np

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# 1.1 Imports
from tflite_runtime.interpreter import Interpreter

# 1.2 Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# 1.3 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
def preprocess_frame(frame, input_shape):
    """Preprocess a frame for the model."""
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    return image.astype(input_details[0]['dtype'])

# Open video file
cap = cv2.VideoCapture(input_path)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Phase 2.2 Preprocess Data
    input_data = preprocess_frame(frame, input_details[0]['shape'])
    
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
            (left, right, top, bottom) = (xmin * cap.get(3), xmax * cap.get(3),
                                          ymin * cap.get(4), ymax * cap.get(4))
            
            # Draw bounding box and label on frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()

print("Video processing complete.")