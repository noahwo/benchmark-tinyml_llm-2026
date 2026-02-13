import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import os

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

def preprocess_input(image, input_details):
    # Resize and normalize the image
    size = (input_details[0]['shape'][2], input_details[0]['shape'][1])
    resized_image = cv2.resize(image, size)
    normalized_image = np.expand_dims(resized_image / 255.0, axis=0).astype(np.float32)
    
    # Convert to UINT8 as required by the model
    return (normalized_image * 255).astype(np.uint8)

def postprocess_output(interpreter, scores, boxes, classes, input_shape):
    height, width = input_shape[:2]
    num_detections = int(scores.shape[1])
    detections = []
    
    for i in range(num_detections):
        if scores[0][i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            detections.append((int(left), int(top), int(right), int(bottom),
                               scores[0][i], classes[0][i]))
    return detections

def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize the TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open video files
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the input image
        input_data = preprocess_input(frame, input_details)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensors
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])

        # Postprocess the results
        detections = postprocess_output(interpreter, scores, boxes, classes, frame.shape)

        # Draw detection boxes and labels on the frame
        for (left, top, right, bottom, score, class_id) in detections:
            label = f"{labels[int(class_id)]}: {score:.2f}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame into the file
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()