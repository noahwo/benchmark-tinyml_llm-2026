import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Error opening video file")

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_shape = input_details[0]['shape']
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))

    # Run inference
    interpreter.invoke()

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Loop over all detections and draw on the frame
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            ymin = int(max(1, (boxes[0][i][0] * frame_height)))
            xmin = int(max(1, (boxes[0][i][1] * frame_width)))
            ymax = int(min(frame_height, (boxes[0][i][2] * frame_height)))
            xmax = int(min(frame_width, (boxes[0][i][3] * frame_width)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            label = labels[int(classes[0][i])]
            score = scores[0][i]
            display_text = f"{label}: {score:.2f}"
            cv2.putText(frame, display_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)

    # Write the frame with detection results to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()