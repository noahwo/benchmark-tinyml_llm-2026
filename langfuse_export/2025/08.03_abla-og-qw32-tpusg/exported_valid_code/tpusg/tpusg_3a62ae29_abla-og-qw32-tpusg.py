import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path      = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path      = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path      = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path     = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold     = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load EdgeTPU delegate: {e}")
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load EdgeTPU delegate from alternative path: {e}")
        raise

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)

# Input Acquisition & Preprocessing Loop
import cv2  # Importing cv2 within the function to avoid unnecessary imports if not needed

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Get the first element as it is a batch
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Ensure we get a scalar value

    for i in range(num_detections):
        confidence = float(scores[i])
        if confidence > confidence_threshold:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            h, w = frame.shape[:2]
            y_min, x_min, y_max, x_max = int(y_min * h), int(x_min * w), int(y_max * h), int(x_max * w)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()