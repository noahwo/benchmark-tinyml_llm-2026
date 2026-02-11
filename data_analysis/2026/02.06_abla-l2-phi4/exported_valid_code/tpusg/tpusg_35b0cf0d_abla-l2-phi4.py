import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

def load_labels(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    interpreter.set_tensor(tensor_index, image)

def get_output(interpreter, score_threshold):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]  # Confidence of detected objects
    
    results = []
    for i in range(len(scores)):
        if scores[i] >= score_threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def draw_detections(image, detections, labels):
    for detection in detections:
        ymin, xmin, ymax, xmax = detection['bounding_box']
        class_id = int(detection['class_id'])
        score = detection['score']

        left, right, top, bottom = int(xmin * image.shape[1]), int(xmax * image.shape[1]), int(ymin * image.shape[0]), int(ymax * image.shape[0])
        
        cv2.rectangle(image, (left, top), (right, bottom), color=(0, 255, 0), thickness=2)
        label = f"{labels[class_id]}: {int(score * 100)}%"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
    label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
    input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
    output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    labels = load_labels(label_path)

    # Setup
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    input_size = tuple(interpreter.get_input_details()[0]['shape'][1:3])

    # Preprocessing
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        set_input_tensor(interpreter, frame)
        interpreter.invoke()

        detections = get_output(interpreter, confidence_threshold)

        draw_detections(frame, detections, labels)
        
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()