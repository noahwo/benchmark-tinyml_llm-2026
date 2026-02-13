import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    # Normalize to [0, 255] and convert to UINT8
    input_tensor = image.astype(np.uint8)
    interpreter.set_tensor(tensor_index, input_tensor)

def get_output(interpreter, score_threshold):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]

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
        bbox = detection['bounding_box']
        class_id = int(detection['class_id'])
        score = detection['score']

        # Denormalize bounding box
        ymin, xmin, ymax, xmax = bbox
        (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                      ymin * image.shape[0], ymax * image.shape[0])

        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)),
                      (10, 255, 0), thickness=4)
        
        label = f"{labels[class_id]}: {int(score * 100)}%"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(top, label_size[1])
        cv2.rectangle(image, (int(left), int(top - round(1.5 * label_size[1]))),
                      (int(left + round(1.5 * label_size[0])), int(top + base_line)),
                      (255, 255, 255), cv2.FILLED)
        
        cv2.putText(image, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), thickness=2)

def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize interpreter with EdgeTPU delegate
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open video files
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Set input tensor
        set_input_tensor(interpreter, frame)

        # Run inference
        interpreter.invoke()

        # Get results
        detections = get_output(interpreter, confidence_threshold)

        # Draw detection boxes with labels
        draw_detections(frame, detections, labels)

        # Write the frame with detections to output video file
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()