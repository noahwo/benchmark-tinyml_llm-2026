import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

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
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input size expected by the model
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Read video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match the model's expected input size
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Set tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))

    # Run inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Loop over detections and draw bounding boxes if confidence > threshold
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            class_id = int(classes[0][i])
            label = labels[class_id]
            box = boxes[0][i]

            # Convert normalized coordinates to frame dimensions
            ymin, xmin, ymax, xmax = box
            left = int(xmin * frame_width)
            top = int(ymin * frame_height)
            right = int(xmax * frame_width)
            bottom = int(ymax * frame_height)

            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {int(scores[0][i]*100)}%', (left, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()