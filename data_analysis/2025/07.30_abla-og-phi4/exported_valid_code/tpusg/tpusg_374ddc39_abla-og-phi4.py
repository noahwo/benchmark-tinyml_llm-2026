import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate
import numpy as np

# CONFIGURATION PARAMETERS
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

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
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

floating_model = input_details[0]['dtype'] == np.float32

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(frame_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    
    if floating_model:
        input_data = np.float32(input_data) - 127.5
        input_data /= 127.5

    input_data = input_data[np.newaxis, ...]

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f"{object_name}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Write the frame with detections to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()