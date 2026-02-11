import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    return np.squeeze(interpreter.get_tensor(output_details['index']))

def detect_objects(interpreter, image, threshold):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def draw_boxes(image, results, labels):
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * image.shape[1])
        xmax = int(xmax * image.shape[1])
        ymin = int(ymin * image.shape[0])
        ymax = int(ymax * image.shape[0])

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        
        label = f'{labels[int(obj["class_id"])]}: {int(obj["score"] * 100)}%'
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(ymin, label_size[1])
        cv2.rectangle(image, (xmin, top - round(1.5*label_size[1])), 
                              (xmin + round(1.5*label_size[0]), top + base_line), 
                              (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (xmin, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

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

    height, width = input_details[0]['shape'][1:3]

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.resize(frame, (width, height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = detect_objects(interpreter, image_rgb, confidence_threshold)
        draw_boxes(frame, results, labels)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    main()