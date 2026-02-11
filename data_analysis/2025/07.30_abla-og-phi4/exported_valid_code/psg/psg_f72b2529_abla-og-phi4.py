import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
num_threads = 2
conf_threshold = 0.5

# Phase 1: Setup
# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter and Get Model Details
interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Video Capture Setup
cap = cv2.VideoCapture(input_path)

# Output Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    # Phase 2: Input Acquisition & Preprocessing Loop
    ret, frame = cap.read()
    if not ret:
        break

    # Resize image to match model's expected size
    resized_frame = cv2.resize(frame, (300, 300))

    # Convert the color from BGR to RGB
    input_data = np.expand_dims(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), axis=0)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] > conf_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * cap.get(3), xmax * cap.get(3),
                                          ymin * cap.get(4), ymax * cap.get(4))
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*label_size[1]))),
                          (int(left + round(1.5*label_size[0])), int(top + base_line)),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

    # Display the resulting frame
    out.write(frame)  # Write the frame with detected objects to output video

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()