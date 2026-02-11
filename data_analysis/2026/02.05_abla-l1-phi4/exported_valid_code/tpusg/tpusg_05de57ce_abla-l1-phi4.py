import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

def load_labels(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    interpreter.tensor(tensor_index)()[...] = image

def get_output(interpreter):
    boxes = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])[0]
    classes = interpreter.get_tensor(
        interpreter.get_output_details()[1]['index'])[0]
    scores = interpreter.get_tensor(
        interpreter.get_output_details()[2]['index'])[0]
    count = int(interpreter.get_tensor(
        interpreter.get_output_details()[3]['index'])[0])
    return boxes, classes, scores, count

def detect_objects(video_path, output_path, model_path, label_path, confidence_threshold):
    # Load labels
    labels = load_labels(label_path)
    
    # Initialize TFLite interpreter with EdgeTPU delegate
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()
    
    input_size = tuple(interpreter.get_input_details()[0]['shape'][1:3])
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Set the input tensor
        set_input_tensor(interpreter, frame)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensors
        boxes, classes, scores, count = get_output(interpreter)
        
        for i in range(count):
            if scores[i] >= confidence_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * width, xmax * width,
                                              ymin * height, ymax * height)
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                label = f'{labels[int(classes[i])]}: {scores[i]:.2f}'
                cv2.putText(frame, label, (int(left), int(top) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Write the frame with detections
        out.write(frame)
    
    cap.release()
    out.release()

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Run object detection
detect_objects(input_path, output_path, model_path, label_path, confidence_threshold)