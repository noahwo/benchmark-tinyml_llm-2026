import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels from the label map
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(frame):
    # Resize the frame to match the input size of the model
    _, height, width, _ = input_details[0]['shape']
    image_resized = cv2.resize(frame, (width, height))
    return np.expand_dims(image_resized, axis=0)

def postprocess(output_data, frame, threshold):
    # Extract output data
    boxes = output_data[0][0]  # Bounding box coordinates
    classes = output_data[1][0]  # Class indices
    scores = output_data[2][0]  # Confidence scores

    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            # Scale bounding boxes back to original image size
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = (xmin * width, xmax * width,
                                        ymin * height, ymax * height)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),
                          (0, 255, 0), 2)
            class_id = int(classes[i])
            label = labels[class_id]
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # Open input and output video files
    cap = cv2.VideoCapture(input_path)
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the image
        input_data = preprocess_image(frame)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output data
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])

        # Process the results and draw on the frame
        postprocess([boxes, classes, scores], frame, confidence_threshold)

        # Write the processed frame to the output video file
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        
        out.write(frame)

    # Release resources
    cap.release()
    if out is not None:
        out.release()

if __name__ == "__main__":
    main()