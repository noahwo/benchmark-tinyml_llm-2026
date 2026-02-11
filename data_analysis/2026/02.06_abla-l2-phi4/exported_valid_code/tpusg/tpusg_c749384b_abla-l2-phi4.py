import os
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from label file
def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(label_path)

# Initialize TFLite interpreter
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0")]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']

# Open video files
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(img_resized, axis=0)

    # Set tensor type to UINT8 for the input
    input_data_uint8 = input_data.astype(np.uint8)
    interpreter.set_tensor(input_index, input_data_uint8)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            class_id = int(classes[i])
            label = f'{labels[class_id]}: {int(scores[i] * 100)}%'
            
            # Draw a bounding box rectangle and label on the image
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(int(top), label_size[1])
            cv2.rectangle(frame, (int(left), int(top - label_size[1])), 
                          (int(left + label_size[0]), int(top + base_line)), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with detection boxes
    out.write(frame)
    print(f"Processed frame in {end_time - start_time:.4f} seconds")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved to:", output_path)