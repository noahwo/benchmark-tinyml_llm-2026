import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
video_path = 'path_to_video.mp4'

# Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Setup Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details  = "data/object_detection/sheeps.mp4"
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

# Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input frame for the model
    input_shape = input_details[0]['shape']
    input_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_frame = np.expand_dims(input_frame, axis=0)
    input_frame = (np.float32(input_frame) - 127.5) / 127.5

    # Set Input Tensor
    interpreter.set_tensor(input_details[0]['index'], input_frame)

    # Run Inference
    interpreter.invoke()

    # Get Output Tensors
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret Results (assuming SSD model with bounding boxes, scores, and classes)
    num_detections = int(output_data[0][0])
    for i in range(num_detections):
        if output_data[1][0][i] < 0.5:  # Use a threshold to filter detections
            continue

        ymin, xmin, ymax, xmax = output_data[2][0][i]
        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                      ymin * frame.shape[0], ymax * frame.shape[0])

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        object_class = labels[int(output_data[3][0][i])]
        label = f'{object_class}: {output_data[1][0][i]:.2f}'
        cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()