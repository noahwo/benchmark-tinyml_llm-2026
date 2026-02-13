import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate
import time

# Configuration Parameters
model_path   = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path   = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path   = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load labels
def load_labels(path):
    with open(path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Initialize interpreter
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e:
    print(f"Failed to load EdgeTPU delegate: {e}")
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        raise RuntimeError("Failed to initialize TFLite interpreter with EdgeTPU.")

interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine the expected data type of the model's input
expected_dtype = input_details[0]['dtype']

# Video capture and writing setup
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError(f"Cannot open video file {input_path}")

width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_frame = np.expand_dims(input_frame, axis=0)

    if expected_dtype == np.uint8:
        # If model expects UINT8, ensure input is in that range
        input_frame = input_frame.astype(np.uint8)
    elif expected_dtype == np.float32:
        # Normalize the image to [-1, 1] as expected by many models
        input_frame = (np.float32(input_frame) - 127.5) / 127.5

    # Set model input and run inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Get output details
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write the frame with detections
    out.write(frame)
    
    print(f"Inference time: {inference_time:.4f} seconds")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()