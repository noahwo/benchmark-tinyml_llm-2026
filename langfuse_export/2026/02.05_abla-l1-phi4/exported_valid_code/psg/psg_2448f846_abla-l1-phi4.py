import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    input_data = np.expand_dims(image, axis=0)
    interpreter.set_tensor(tensor_index, input_data)

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

def draw_objects(frame, results, labels):
    height, width, _ = frame.shape
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
        cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), thickness=4)
        label = f"{labels[int(obj['class_id'])]} {int(obj['score'] * 100)}%"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(top, label_size[1])
        cv2.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)

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
    output_details = interpreter.get_output_details()
    _, input_height, input_width, _ = input_details[0]['shape']

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (input_width, input_height))
        set_input_tensor(interpreter, frame_resized)

        interpreter.invoke()
        results = get_output(interpreter, confidence_threshold)

        draw_objects(frame, results, labels)
        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    main()