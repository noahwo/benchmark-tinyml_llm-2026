import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

def load_labels(label_path):
    """Load the labels from a file."""
    with open(label_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    """Preprocess the input image and set it to the interpreter."""
    # Get input details
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    
    # Resize and normalize the image
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    image = (image / 255.0).astype(np.float32)

    # Set the tensor to the interpreter
    interpreter.set_tensor(input_details[0]['index'], image.astype(np.uint8))

def get_output(interpreter):
    """Get detection results from the interpreter."""
    output_details = interpreter.get_output_details()
    
    # Get all outputs from the model
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    
    return boxes, classes, scores

def draw_detections(image, boxes, classes, scores, labels, threshold):
    """Draw bounding boxes and labels on the image."""
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                          ymin * image.shape[0], ymax * image.shape[0])
            
            # Draw rectangle and label
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i]*100)}%"
            cv2.putText(image, label, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

def main():
    # Configuration parameters
    model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
    label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
    input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
    output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    # Load labels
    labels = load_labels(label_path)

    # Initialize the TFLite interpreter with EdgeTPU delegate
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    # Open video files
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Set input tensor
        set_input_tensor(interpreter, frame)

        # Run inference
        interpreter.invoke()

        # Get output and draw detections
        boxes, classes, scores = get_output(interpreter)
        draw_detections(frame, boxes, classes, scores, labels, confidence_threshold)

        # Write the frame with detections to the output video
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()