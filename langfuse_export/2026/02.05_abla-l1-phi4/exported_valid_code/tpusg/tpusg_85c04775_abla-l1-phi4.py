import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Initialize TFLite interpreter with EdgeTPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

# Open video file
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    elapsed_ms = (time.time() - start_time) * 1000

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])

            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]
            label = f"{object_name}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Write frame to output video
    out.write(frame)
    print(f"Inference time: {elapsed_ms:.2f} ms")

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()