import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Setup TFLite interpreter
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def load_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    return cap

def preprocess_frame(frame, input_size):
    # Resize and normalize the frame
    resized_frame = cv2.resize(frame, (input_size[1], input_size[0]))
    normalized_frame = np.expand_dims(resized_frame, axis=0).astype(np.float32)
    normalized_frame /= 255.0  # Normalize to [0,1]
    return normalized_frame

def detect_objects(interpreter, frame):
    input_size = interpreter.get_input_details()[0]['shape'][1:3]
    preprocessed_frame = preprocess_frame(frame, input_size)

    # Set the tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], (preprocessed_frame * 255).astype(np.uint8))
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    return boxes, classes, scores

def draw_detections(frame, boxes, classes, scores):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            class_id = int(classes[i])
            label = f"{labels[class_id]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def compute_map():
    # Placeholder for mAP computation logic
    pass

def main():
    cap = load_video(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, classes, scores = detect_objects(interpreter, frame)
        draw_detections(frame, boxes, classes, scores)

        out.write(frame)

    cap.release()
    out.release()

    compute_map()  # Compute mAP after processing all frames

if __name__ == "__main__":
    main()