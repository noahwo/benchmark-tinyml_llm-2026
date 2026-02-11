import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image, axis=0)
    interpreter.set_tensor(tensor_index, input_data)

def get_output(interpreter):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]
    count = int(interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0])
    return boxes, classes, scores, count

def detect_objects(model_path, label_path, input_path, output_path, confidence_threshold):
    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Load labels
    labels = load_labels(label_path)

    # Open video files
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Set input tensor
        set_input_tensor(interpreter, frame)

        # Run inference
        interpreter.invoke()

        # Get output tensors
        boxes, classes, scores, count = get_output(interpreter)

        for i in range(count):
            if scores[i] >= confidence_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                              ymin * frame_height, ymax * frame_height)
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),
                              (10, 255, 0), 2)
                label = f'{labels[int(classes[i])]}: {int(scores[i] * 100)}%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                top = max(top, label_size[1])
                cv2.rectangle(frame, (int(left), int(top - round(1.5 * label_size[1]))),
                              (int(left + round(1.5 * label_size[0])), int(top + base_line)),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 2)

        # Write frame with detections
        out.write(frame)

    cap.release()
    out.release()

# Configuration parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Run object detection
detect_objects(model_path, label_path, input_path, output_path, confidence_threshold)