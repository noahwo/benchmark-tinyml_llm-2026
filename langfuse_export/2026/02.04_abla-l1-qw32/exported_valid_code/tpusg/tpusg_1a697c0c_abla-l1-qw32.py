import cv2
from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = {index: line.strip() for index, line in enumerate(f.readlines())}

# Initialize interpreter with Edge TPU delegate
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read video from file
cap = cv2.VideoCapture(input_path)

# Get video properties to set up the writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and resize
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_shape = input_details[0]['shape']
    resized_frame = Image.fromarray(rgb_frame).resize((input_shape[1], input_shape[2]), Image.ANTIALIAS)

    # Prepare input tensor
    input_tensor = np.array(resized_frame).reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw detections on the frame
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            h, w, _ = frame.shape
            (xminn, xmaxx, yminn, ymaxx) = (int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h))
            label = labels[int(classes[i])]
            cv2.rectangle(frame, (xminn, yminn), (xmaxx, ymaxx), (10, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (xminn, yminn - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()