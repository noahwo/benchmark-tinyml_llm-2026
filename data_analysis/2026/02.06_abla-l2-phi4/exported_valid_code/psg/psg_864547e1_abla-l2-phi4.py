import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims(image, 0)

def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def draw_box_on_image(num, box, class_name, score, frame_width, frame_height):
    ymin, xmin, ymax, xmax = box
    left, right, top, bottom = int(xmin * frame_width), int(xmax * frame_width), int(ymin * frame_height), int(ymax * frame_height)
    
    cv2.rectangle(frame, (left, top), (right, bottom), color=(0, 255, 0), thickness=3)
    label = f'{class_name}: {score:.2f}'
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    global frame

    labels = load_labels(label_path)
    
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.resize(frame, (input_width, input_height))
        set_input_tensor(interpreter, image)

        interpreter.invoke()

        boxes = get_output_tensor(interpreter, 0)
        classes = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)

        count = int(get_output_tensor(interpreter, 3))

        for i in range(count):
            if scores[i] >= confidence_threshold:
                draw_box_on_image(i, boxes[i], labels[int(classes[i])], scores[i], width, height)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()