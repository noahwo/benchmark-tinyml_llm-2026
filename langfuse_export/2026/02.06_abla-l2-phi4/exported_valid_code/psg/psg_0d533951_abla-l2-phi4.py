import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Video capture and writer setup
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Function to preprocess the input image
def preprocess_image(frame):
    # Resize and normalize the frame
    img = cv2.resize(frame, (300, 300))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img = img / 255.0  # Normalize to [0,1]
    return img.astype(np.uint8)  # Convert to UINT8 as expected by the model

# Function to draw bounding boxes and labels
def draw_boxes(frame, detections):
    for detection in detections:
        # Extract information from the detection
        ymin, xmin, ymax, xmax = detection[0:4]
        score = detection[2]
        class_id = int(detection[1])

        if score >= confidence_threshold:
            # Convert normalized coordinates to pixel values
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            
            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            label = f"{labels[class_id]}: {score:.2f}"
            cv2.putText(frame, label, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image for model input
    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects

    # Combine results into a single list for easier processing
    detections = np.column_stack((boxes, classes, scores))

    # Draw bounding boxes and labels on the frame
    draw_boxes(frame, detections)

    # Write the processed frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()