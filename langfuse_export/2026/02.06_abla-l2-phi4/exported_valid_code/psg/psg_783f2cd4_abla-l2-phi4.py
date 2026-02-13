import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open input video
cap = cv2.VideoCapture(input_path)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input image
    height, width, _ = frame.shape
    input_frame = cv2.resize(frame, (300, 300))
    input_frame = np.expand_dims(input_frame, axis=0)

    # Convert to UINT8 as expected by the model
    input_frame = np.uint8(255 * (input_frame / 255.0))

    # Set tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence score

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)

            label = labels[int(classes[i])]
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Note: mAP calculation is not implemented here as it requires ground truth annotations and additional logic.