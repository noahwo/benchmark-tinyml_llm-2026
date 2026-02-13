import cv2
import numpy as np
import os

from tflite_runtime.interpreter import Interpreter

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def preprocess_input(image, input_size):
    image = cv2.resize(image, (input_size[1], input_size[0]))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.uint8)  # Ensure the data type is UINT8
    return image

def draw_detections(frame, detections, labels, threshold):
    height, width, _ = frame.shape
    for detection in detections[0][0]:
        score = detection[2]
        if score > threshold:
            bbox = detection[3:7] * np.array([width, height, width, height])
            left, top, right, bottom = bbox.astype(int)
            class_id = int(detection[1])
            label = labels[class_id]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            text = f"{label}: {score:.2f}"
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    # Configuration
    model_path = "models/ssd-mobilenet_v1/detect.tflite"
    label_path = "models/ssd-mobilenet_v1/labelmap.txt"
    input_path = "data/object_detection/sheeps.mp4"
    output_path = "results/object_detection/test_results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, height, width, _ = input_details[0]['shape']

    # Open video files
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess input
        input_data = preprocess_input(frame, (height, width))

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get detections
        detections = [interpreter.get_tensor(output['index']) for output in output_details]

        # Draw detections on the frame
        draw_detections(frame, detections, labels, confidence_threshold)

        # Write the frame with detections
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()