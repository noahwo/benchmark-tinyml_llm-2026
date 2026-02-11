import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def make_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    return interpreter.get_input_details()

def get_output_details(interpreter):
    return interpreter.get_output_details()

def set_input_tensor(interpreter, image):
    tensor_index = get_input_details(interpreter)[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output(interpreter, score_threshold):
    boxes = interpreter.get_tensor(get_output_details(interpreter)[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(get_output_details(interpreter)[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(get_output_details(interpreter)[2]['index'])[0]  # Confidence of detected objects

    results = []
    for i in range(len(scores)):
        if scores[i] >= score_threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            }
            results.append(result)
    return results

def load_image_pixels(frame, input_size):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_size, input_size))
    input_data = np.expand_dims(image_resized, axis=0)
    return input_data

def draw_results(frame, results, labels):
    for result in results:
        ymin, xmin, ymax, xmax = result['bounding_box']
        class_id = int(result['class_id'])
        score = result['score']

        left = int(xmin * frame.shape[1])
        right = int(xmax * frame.shape[1])
        top = int(ymin * frame.shape[0])
        bottom = int(ymax * frame.shape[0])

        cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), 2)
        
        label = f"{labels[class_id]}: {int(score * 100)}%"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_offset_x = left
        text_offset_y = top - 7 if top - 7 > text_size[1] else top + text_size[1]

        cv2.rectangle(frame, (text_offset_x, text_offset_y - text_size[1]), 
                      (text_offset_x + text_size[0], text_offset_y + 5), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 10, 255), 1)

def main():
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = 0.5

    labels = load_labels(label_path)
    interpreter = make_interpreter(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    input_details = get_input_details(interpreter)
    output_details = get_output_details(interpreter)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_data = load_image_pixels(frame, input_details[0]['shape'][2])
        set_input_tensor(interpreter, input_data)
        
        interpreter.invoke()

        results = get_output(interpreter, confidence_threshold)

        draw_results(frame, results, labels)

        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()