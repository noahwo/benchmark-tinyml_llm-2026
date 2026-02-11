import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def load_model():
    # Load TFLite model and allocate tensors with EdgeTPU delegate
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame, input_size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_size[1], input_size[0]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32)
    return input_data

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_boxes(interpreter, score_threshold):
    boxes = interpreter.get_output_details()[0]['index']
    classes = interpreter.get_output_details()[1]['index']
    scores = interpreter.get_output_details()[2]['index']

    output_data = [
        np.squeeze(interpreter.get_tensor(boxes)),
        np.squeeze(interpreter.get_tensor(classes)).astype(np.int32),
        np.squeeze(interpreter.get_tensor(scores))
    ]

    detection_boxes = []
    for i in range(len(output_data[0])):
        if output_data[2][i] >= score_threshold:
            box = tuple(output_data[0][i].tolist())
            class_id = int(output_data[1][i])
            score = float(output_data[2][i])
            detection_boxes.append((box, class_id, score))
    return detection_boxes

def draw_boxes(frame, boxes):
    for box, class_id, score in boxes:
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                     ymin * frame.shape[0], ymax * frame.shape[0])
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
        label = f"{labels[class_id]}: {int(score*100)}%"
        cv2.putText(frame, label, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # Setup
    interpreter = load_model()
    input_size = tuple(interpreter.get_input_details()[0]['shape'][1:3])

    # Video capture and writer setup
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing
        input_data = preprocess_frame(frame, input_size)

        # Inference
        set_input_tensor(interpreter, input_data)
        interpreter.invoke()

        # Output handling
        boxes = get_output_boxes(interpreter, confidence_threshold)
        draw_boxes(frame, boxes)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()