import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = np.expand_dims(image, axis=0)
    interpreter.set_tensor(tensor_index, input_tensor)

def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    return np.squeeze(interpreter.get_tensor(output_details['index']))

def detect_objects(interpreter, image, threshold):
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)

    results = []
    for i in range(scores.size):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                          ymin * image.shape[0], ymax * image.shape[0])
            results.append((int(left), int(top), int(right), int(bottom), scores[i], classes[i]))
    return results

def draw_boxes(image, results, labels):
    for box in results:
        left, top, right, bottom, score, class_id = box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f'{labels[int(class_id)]}: {int(score * 100)}%'
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = 0.5

    labels = load_labels(label_path)

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    _, height, width, _ = input_details[0]['shape']

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = cv2.resize(frame, (width, height))
        results = detect_objects(interpreter, input_frame, confidence_threshold)
        draw_boxes(frame, results, labels)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    main()