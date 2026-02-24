import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def load_model():
    # Initialize the TFLite interpreter with EdgeTPU delegate
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame, input_details):
    # Resize and normalize the frame to match the model's expected input
    input_shape = input_details[0]['shape']
    image_size = (input_shape[1], input_shape[2])
    frame_resized = cv2.resize(frame, image_size)
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
    return input_data

def infer(interpreter, input_data):
    # Set the tensor to be used for inference
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    return boxes, classes, scores

def draw_detections(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

def main():
    # Load the TFLite model and allocate tensors
    interpreter = load_model()

    global input_details, output_details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_data = preprocess_frame(frame, input_details)

        # Perform inference
        boxes, classes, scores = infer(interpreter, input_data)

        # Draw detections on the frame
        draw_detections(frame, boxes, classes, scores)

        # Write the frame with detections to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    main()