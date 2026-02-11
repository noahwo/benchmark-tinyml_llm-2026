import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import cv2

# Phase 1: Setup
## Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

## Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

## Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
interpreter.allocate_tensors()

## Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = (input_details[0]['dtype'] == np.float32)
height, width = input_details[0]['shape'][1:3]

# Phase 2: Input Acquisition & Preprocessing Loop
## Acquire Input Data
cap = cv2.VideoCapture(input_path)
video_writer = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    ## Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    
    ## Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    ## Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
    
    ## Interpret Results and Post-processing
    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            
            ## Draw bounding box and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    ## Handle Output
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                       (frame.shape[1], frame.shape[0]))
    
    video_writer.write(frame)

# Phase 5: Cleanup
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()