import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Paths and Parameters
model_path    = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path    = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path    = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path   = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Error loading libedgetpu.so.1.0: {e}")
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Error loading /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0: {e}")
        raise

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]
floating_model = (input_details[0]['dtype'] == np.float32)

# Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run Inference
    interpreter.invoke()

    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Interpret Results
    for i in range(num_detections):
        if scores[0][i] > confidence_threshold:  # Corrected indexing here
            ymin, xmin, ymax, xmax = boxes[0][i]
            label = labels[int(classes[0][i])]  # Corrected indexing here
            score = scores[0][i]  # Corrected indexing here

            ymin = int(max(1, ymin * frame_height))
            xmin = int(max(1, xmin * frame_width))
            ymax = int(min(frame_height, ymax * frame_height))
            xmax = int(min(frame_width, xmax * frame_width))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Handle Output
    out.write(frame)

# Cleanup
cap.release()
out.release()

print("Video processing complete.")