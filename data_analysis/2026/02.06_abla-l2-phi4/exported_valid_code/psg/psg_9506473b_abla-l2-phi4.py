import cv2
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']

    # Resize the image to fit the model expected size and convert to UINT8
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    
    input_tensor = np.asarray(image, dtype=np.uint8)

    interpreter.set_tensor(tensor_index, input_tensor)

def get_output(interpreter, score_threshold):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0] # Confidence of detected objects
    
    # Filter out boxes with low confidence
    n = len(scores)
    for i in range(n):
        if scores[i] >= score_threshold:
            yield scores[i], classes[i], boxes[i]

def draw_box_on_image(num, class_id, box, im_width, im_height, image, label):
    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    
    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), thickness=4)

    # Label
    label = '{}: {}%'.format(label, int(100 * num))
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top = max(top, label_size[1])
    cv2.putText(image, label, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)

def main():
    model_path = "models/ssd-mobilenet_v1/detect.tflite"
    label_path = "models/ssd-mobilenet_v1/labelmap.txt"
    input_path = "data/object_detection/sheeps.mp4"
    output_path = "results/object_detection/test_results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    labels = load_labels(label_path)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    output_height = interpreter.get_output_details()[0]['shape'][1]
    output_width = interpreter.get_output_details()[0]['shape'][2]

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        set_input_tensor(interpreter, frame)
        interpreter.invoke()

        for num, class_id, box in get_output(interpreter, confidence_threshold):
            draw_box_on_image(num, int(class_id), box, width, height, frame, labels[int(class_id)])

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    main()