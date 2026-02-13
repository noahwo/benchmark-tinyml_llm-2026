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

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if the input tensor is quantized
floating_model = (input_details[0]['dtype'] == np.float32)

# Load video capture
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (note: only for float models)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Run the inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Loop over all detections and draw boxes around detected objects above the threshold
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = map(int, (boxes[i][0] * height, boxes[i][1] * width, 
                                                boxes[i][2] * height, boxes[i][3] * width))
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write the frame with detection to output video
    out.write(frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()