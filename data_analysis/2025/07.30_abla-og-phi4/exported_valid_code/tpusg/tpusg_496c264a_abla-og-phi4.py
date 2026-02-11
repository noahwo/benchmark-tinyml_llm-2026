import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Load labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Load interpreter with EdgeTPU delegate
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except ValueError:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and pad the image to fit model input size
    img_size = np.array([width, height], dtype=np.float32)
    bbox_scale = min(width / max(frame.shape[1], 1), height / max(frame.shape[0], 1))
    scaled_img = cv2.resize(frame, (0, 0), fx=bbox_scale, fy=bbox_scale)

    pad_w = int((width - scaled_img.shape[1]) / 2)
    pad_h = int((height - scaled_img.shape[0]) / 2)
    padded_img = cv2.copyMakeBorder(scaled_img, pad_h, height - scaled_img.shape[0] - pad_h,
                                    pad_w, width - scaled_img.shape[1] - pad_w, cv2.BORDER_CONSTANT)

    input_data = np.expand_dims(padded_img, axis=0).astype(np.uint8)
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] < confidence_threshold:
            continue

        # Scale boxes back to original image size
        ymin, xmin, ymax, xmax = boxes[i]
        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                      ymin * frame.shape[0], ymax * frame.shape[0])
        
        # Clip the bounding box coordinates to be within the image dimensions
        left, right = max(0, int(left)), min(frame.shape[1], int(right))
        top, bottom = max(0, int(top)), min(frame.shape[0], int(bottom))

        # Draw bounding box and label on frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f'{labels[int(classes[i])]}: {int(scores[i] * 100)}%'
        cv2.putText(frame, label, (left, max(top - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write the frame with detections to output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# Phase 5: Cleanup
print("Processing complete.")