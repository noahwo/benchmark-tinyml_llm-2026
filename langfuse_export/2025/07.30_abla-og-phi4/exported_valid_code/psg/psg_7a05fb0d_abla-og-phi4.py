from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details  = "data/object_detection/sheeps.mp4"
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Phase 2: Input Acquisition & Preprocessing Loop
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assuming the model outputs a list of class indices with confidence scores
    for i in range(len(labels)):
        if output_data[0][i] > some_threshold:  # Define your threshold
            print(f"Detected {labels[i]} with confidence {output_data[0][i]}")

# Phase 5: Cleanup
cap.release()
cv2.destroyAllWindows()