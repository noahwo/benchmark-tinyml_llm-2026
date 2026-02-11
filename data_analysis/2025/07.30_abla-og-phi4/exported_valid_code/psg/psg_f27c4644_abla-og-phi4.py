# Phase 1: Setup

# Import necessary libraries
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

# Configuration parameters
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_MAP_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"

# Load labels if necessary (e.g., for classification tasks)
with open(LABEL_MAP_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details from the model
input_details  = "data/object_detection/sheeps.mp4"
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 2: Input Acquisition & Preprocessing Loop

# Define input path for video file or camera stream
VIDEO_INPUT_PATH = 'path_to_video.mp4'  # Update with your actual video path

# Open a video capture object
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the input frame to match model's expected input shape and type
    input_shape = input_details[0]['shape']
    input_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_frame = np.expand_dims(input_frame, axis=0)
    
    if input_details[0]['dtype'] == np.uint8:
        input_frame = input_frame.astype(np.uint8)
    else:
        input_frame = (np.float32(input_frame) - 127.5) / 127.5

    # Phase 3: Inference
    
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Example of interpreting the results (assuming classification task)
    # This part may vary based on your specific model's output
    if len(output_data.shape) == 2:
        probabilities = np.squeeze(output_data)
        top_prediction_index = np.argmax(probabilities)
        predicted_label = labels[top_prediction_index]
        confidence = probabilities[top_prediction_index]

        print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}")

    # Display the frame (optional)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 5: Cleanup
cap.release()
cv2.destroyAllWindows()