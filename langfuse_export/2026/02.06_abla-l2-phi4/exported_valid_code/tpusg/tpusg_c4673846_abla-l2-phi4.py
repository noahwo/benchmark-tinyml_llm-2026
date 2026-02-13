import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def preprocess_input(image, input_size):
    image_resized = cv2.resize(image, (input_size[1], input_size[0]))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
    return input_data

def draw_boxes(image, boxes, classes, scores, labels, confidence_threshold):
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                          ymin * image.shape[0], ymax * image.shape[0])
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.putText(image, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    # Configuration parameters
    model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
    label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
    input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
    output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    # Load labels
    labels = load_labels(label_path)

    # Initialize the TFLite interpreter with EdgeTPU support
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1:3]

    # Open video files
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess input
        input_data = preprocess_input(frame, input_size)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Extract output data
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # Draw bounding boxes and labels on the frame
        draw_boxes(frame, boxes, classes, scores, labels, confidence_threshold)

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    main()