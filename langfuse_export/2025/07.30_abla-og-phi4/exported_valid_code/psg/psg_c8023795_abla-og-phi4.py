import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_size  = "data/object_detection/sheeps.mp4"
video_path = 'path_to_video_file'  # Replace with actual video path or use 0 for webcam
confidence_threshold  = 0.5

# Load labels
labels = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light"
]

# Phase 1: Setup
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

input_shape = input_details[0]['shape']

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize and normalize the image
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0
    # Add a batch dimension
    return np.expand_dims(normalized_frame, axis=0).astype(np.float32)

# Phase 4.2: Interpret Results function
def interpret_results(output_data):
    num_detections = int(output_data[0][0])
    detection_boxes = output_data[1][0]
    detection_classes = output_data[2][0].astype(int)
    detection_scores = output_data[3][0]

    results = []
    for i in range(num_detections):
        if detection_scores[i] >= confidence_threshold:
            box = tuple(detection_boxes[i].tolist())
            class_id = int(detection_classes[i])
            score = float(detection_scores[i])
            label = labels[class_id]
            results.append((box, label, score))
    return results

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_data = preprocess_frame(frame)
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensors
    output_data = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]
    
    # Interpret results
    detections = interpret_results(output_data)
    
    # Phase 4.3: Handle Output - draw bounding boxes and labels on frame
    h, w, _ = frame.shape
    
    for box, label, score in detections:
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Put label and score
        text = f'{label}: {score:.2f}'
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 5: Cleanup
cap.release()
cv2.destroyAllWindows()