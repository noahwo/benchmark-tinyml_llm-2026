import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

def load_labels(path):
    """Load the labels from a label file."""
    with open(path, 'r', encoding='utf-8') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def preprocess_frame(frame, input_shape):
    """Preprocess the frame to match model's input requirements."""
    image = cv2.resize(frame, (input_shape[1], input_shape[0]))
    image = np.expand_dims(image, axis=0)
    return image

def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    interpreter.set_tensor(
        interpreter.get_input_details()[0]['index'],
        image.astype(np.uint8))

def get_output(interpreter, score_threshold):
    """Return detection results from the model that are above the threshold."""
    boxes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0])
    classes = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0])
    scores = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0])
    
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

def draw_detections(frame, results, labels):
    """Draw the detection boxes with labels on the frame."""
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        bbox_height = (ymax - ymin) * frame.shape[0]
        bbox_width = (xmax - xmin) * frame.shape[1]

        cv2.rectangle(frame,
                      (int(xmin * frame.shape[1]), int(ymin * frame.shape[0])),
                      (int(xmax * frame.shape[1]), int(ymax * frame.shape[0])), 
                      (10, 255, 0), 2)

        label = f'{labels[int(obj["class_id"])]}: {round(obj["score"], 3)}'
        cv2.putText(frame, label, 
                    (int(xmin * frame.shape[1]), int(ymin * frame.shape[0] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
    label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
    input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
    output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter with EdgeTPU
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get video capture
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_data = preprocess_frame(frame, input_details[0]['shape'][1:3])
        
        # Perform inference
        set_input_tensor(interpreter, input_data)
        interpreter.invoke()

        # Get results and draw detections
        results = get_output(interpreter, confidence_threshold)
        draw_detections(frame, results, labels)

        # Write the frame with detections to output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()