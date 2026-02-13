import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Configuration parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load the labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if the input size is fixed
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to save output video with the same settings
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match input size
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))

    # Run the inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Loop over all detections and draw boxes around detected objects
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            label = labels[int(classes[i]) - 1]  # Labels are 1-indexed
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()