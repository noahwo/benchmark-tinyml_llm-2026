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

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video {input_path}")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    resized_frame = cv2.resize(frame, (input_width, input_height))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Set tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))

    # Run inference
    interpreter.invoke()

    # Extract output data from the interpreter
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Postprocess and draw bounding boxes
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            label = labels[int(classes[0][i])]
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[0][i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()