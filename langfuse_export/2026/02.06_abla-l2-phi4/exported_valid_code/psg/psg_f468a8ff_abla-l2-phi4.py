import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(label_file):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_frame(frame, input_size):
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = (frame_resized / 255.0).astype(np.float32)
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])

    # Open video files
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_data = preprocess_frame(frame, input_size)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))

        # Run inference
        interpreter.invoke()

        # Get output tensors
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

        # Draw bounding boxes and labels on the frame
        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                              ymin * frame_height, ymax * frame_height)
                left, right, top, bottom = int(left), int(right), int(top), int(bottom)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with detections
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    main()