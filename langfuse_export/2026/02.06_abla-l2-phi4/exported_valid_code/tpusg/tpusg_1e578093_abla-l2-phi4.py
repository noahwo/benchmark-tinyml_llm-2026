import os
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    """Preprocess the input image and set it to the input tensor."""
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    height, width = input_details[0]['shape'][1:3]
    
    # Resize and normalize the image
    image = cv2.resize(image, (width, height))
    input_data = np.expand_dims(image, axis=0)
    input_data = (input_data / 255.0).astype(np.uint8)  # Normalize to [0, 1] and convert to UINT8
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

def get_output(interpreter):
    """Return detection results from the inference."""
    output_details = interpreter.get_output_details()
    
    boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
    
    return boxes, classes, scores

def draw_detections(image, boxes, classes, scores, labels):
    """Draw bounding box and label on the image."""
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                          ymin * image.shape[0], ymax * image.shape[0])
            
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = '%s: %d%%' % (labels[int(classes[i])], int(scores[i] * 100))
            cv2.putText(image, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

def main():
    # Load labels
    labels = load_labels(label_path)
    
    # Initialize the interpreter with EdgeTPU delegate
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    interpreter.allocate_tensors()
    
    # Open video files
    cap = cv2.VideoCapture(input_path)
    out = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        set_input_tensor(interpreter, frame)
        interpreter.invoke()
        
        boxes, classes, scores = get_output(interpreter)
        draw_detections(frame, boxes, classes, scores, labels)
        
        # Initialize video writer
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        
        out.write(frame)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    main()