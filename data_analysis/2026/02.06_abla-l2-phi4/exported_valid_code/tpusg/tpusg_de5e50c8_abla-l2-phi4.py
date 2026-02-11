import os
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def load_labels(path):
    p = open(path, "rb")
    result = {}
    for idx, pair in enumerate(p.read().decode("utf-8").strip().split("\n")):
        key, val = pair.split(maxsplit=1)
        result[int(key)] = val.strip()
    return result

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    # Resize and normalize the image
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    image = (image / 255.0).astype(np.uint8)  # Ensure UINT8 format for EdgeTPU
    interpreter.set_tensor(tensor_index, image)

def get_output(interpreter, score_threshold):
    boxes = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(
        interpreter.get_output_details()[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(
        interpreter.get_output_details()[2]['index'])[0]  # Confidence of detected objects
    # Filter out detections with low confidence
    valid_detections = []
    for i in range(len(scores)):
        if scores[i] >= score_threshold:
            valid_detections.append((boxes[i], classes[i], scores[i]))
    return valid_detections

def main():
    # Load TFLite model and allocate tensors.
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the image
        set_input_tensor(interpreter, frame)

        # Run inference
        interpreter.invoke()

        # Get results
        detections = get_output(interpreter, confidence_threshold)

        # Draw boxes and labels on the frame
        for detection in detections:
            bbox, class_id, score = detection
            ymin, xmin, ymax, xmax = bbox

            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),
                          (10, 255, 0), 2)

            label = f"{labels[int(class_id)]}: {int(score*100)}%"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.putText(frame, label, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        # Write the frame into the output video file
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()