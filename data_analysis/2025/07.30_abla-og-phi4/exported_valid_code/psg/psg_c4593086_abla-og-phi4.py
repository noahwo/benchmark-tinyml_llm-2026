from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np

# Configuration parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels from file
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_index = input_details[0]['index']

# Video capture and writer setup
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

    # Set model input and invoke inference
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Retrieve output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Threshold can be adjusted
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            object_name = labels[int(classes[i])]
            
            # Draw a bounding box rectangle and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = f"{object_name}: {int(scores[i] * 100)}%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, labelSize[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*labelSize[1]))), 
                          (int(left + round(1.5*labelSize[0])), int(top + baseLine)), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top + baseLine)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with bounding boxes to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()