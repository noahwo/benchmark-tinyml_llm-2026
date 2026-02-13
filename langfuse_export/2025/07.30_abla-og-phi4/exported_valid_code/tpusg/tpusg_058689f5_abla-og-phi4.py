import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Paths/Parameters & Load Labels
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(label_path)

# Phase 1: Load Interpreter with EdgeTPU
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
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

floating_model = input_details[0]['dtype'] == np.float32

# Phase 3 & 4: Inference & Output Interpretation Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run Inference
    interpreter.invoke()

    # Phase 4: Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Phase 4.2: Interpret Results and Post-processing
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            object_name = labels[int(classes[i])] if int(classes[i]) < len(labels) else 'N/A'
            label = f'{object_name}: {int(scores[i]*100)}%'
            cv2.putText(frame, label, (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Phase 4.4: Handle Output
    out.write(frame)
    
# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()