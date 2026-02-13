import cv2
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from file
def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return {i: line.strip() for i, line in enumerate(lines)}

labels = load_labels(label_path)

# Initialize TFLite interpreter for EdgeTPU
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']

# Prepare the video capture and writer objects
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_size = (int(width * 0.5), int(height * 0.5))  # Adjusted for processing

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
    return input_data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Run inference
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5 * label_size[1]))),
                          (int(left + label_size[0]), int(top + base_line)), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Resize frame for output video
    resized_frame = cv2.resize(frame, output_size)
    
    # Write the processed frame to the output video file
    out.write(resized_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. The output is saved at:", output_path)