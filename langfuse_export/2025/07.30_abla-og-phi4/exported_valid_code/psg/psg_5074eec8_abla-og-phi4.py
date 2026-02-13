from tflite_runtime.interpreter import Interpreter
import cv2

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_mean  = "data/object_detection/sheeps.mp4"
input_std = 127.5

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels if needed for interpretation
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Phase 2: Input Acquisition & Preprocessing Loop
video_capture = cv2.VideoCapture('path_to_video.mp4')

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Resize and preprocess the frame
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = (frame_resized - input_mean) / input_std

    # Ensure correct shape: [1, height, width, channels]
    input_data = input_data.astype(input_details[0]['dtype'])
    input_data = input_data[None, ...]

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assuming the model outputs class probabilities, get top prediction
    top_prediction_index = output_data.argmax()
    label = labels[top_prediction_index]

    # Handle output by displaying the frame with predicted label
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 5: Cleanup
video_capture.release()
cv2.destroyAllWindows()