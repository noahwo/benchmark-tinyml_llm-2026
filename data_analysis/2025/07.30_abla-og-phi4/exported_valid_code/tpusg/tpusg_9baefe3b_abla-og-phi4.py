import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters from the prompt
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load EdgeTPU delegate with 'libedgetpu.so.1.0': {e}. Trying fallback path.")
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        print(f"Failed to load EdgeTPU delegate with fallback path: {e}")
        raise

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(input_data, axis=0)

    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Phase 3: Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            label = f"{labels[int(classes[i])]}: {int(scores[i]*100)}%"
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(left), int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()