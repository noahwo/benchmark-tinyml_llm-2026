import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image, axis=0)
    interpreter.set_tensor(tensor_index, input_data)

def get_output(interpreter):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]  # Confidence of detected objects
    return boxes, classes, scores

def detect_objects(video_path, model_path, label_path, output_path, confidence_threshold):
    labels = load_labels(label_path)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        set_input_tensor(interpreter, frame)
        interpreter.invoke()

        boxes, classes, scores = get_output(interpreter)

        for i in range(len(scores)):
            if scores[i] >= confidence_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
                cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = 0.5

    detect_objects(input_path, model_path, label_path, output_path, confidence_threshold)