import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

# Initialize TFLite interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input image size
_, input_height, input_width, _ = input_details[0]['shape']

# Open video files
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (input_width, input_height))
    input_data = np.expand_dims(resized_image, axis=0)

    # Set the tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], np.uint8(input_data))

    interpreter.invoke()

    # Get output details
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects

    # Draw boxes and labels on the image
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])

            # Draw rectangle and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

    # Write the frame with boxes
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()