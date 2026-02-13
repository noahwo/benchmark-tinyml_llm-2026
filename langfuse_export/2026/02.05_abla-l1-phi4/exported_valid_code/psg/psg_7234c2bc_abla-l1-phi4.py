import cv2
import numpy as np
import tensorflow as tf

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_model(model_path):
    return tf.lite.Interpreter(model_path=model_path)

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
    return boxes, classes, scores

def draw_detections(frame, detections, labels, threshold):
    for box, label, score in zip(*detections):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                      ymin * frame.shape[0], ymax * frame.shape[0])
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        label = f'{labels[int(label)]}: {int(score*100)}%'
        cv2.putText(frame, label, (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = 0.5

    labels = load_labels(label_path)
    interpreter = load_model(model_path)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        set_input_tensor(interpreter, frame)
        interpreter.invoke()

        boxes, classes, scores = get_output(interpreter)
        detections = (boxes, classes, scores)

        draw_detections(frame, detections, labels, confidence_threshold)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()