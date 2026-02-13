import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import numpy as np

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels from label file
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the TFLite interpreter with Edge TPU delegate
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height, width = input_details[0]['shape'][1:3]

# Open video files
video_in = cv2.VideoCapture(input_path)
fps = int(video_in.get(cv2.CAP_PROP_FPS))
frame_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def get_output(interpreter):
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects

    return boxes, classes, scores

while video_in.isOpened():
    ret, frame = video_in.read()
    if not ret:
        break

    # Preprocess the image
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes, classes, scores = get_output(interpreter)

    # Draw bounding boxes on the frame
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)

            # Draw a rectangle and label on the image
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i]*100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5*label_size[1]))),
                          (int(left + round(1.5*label_size[0])), int(top + base_line)), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top + base_line)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with bounding boxes
    video_out.write(frame)

video_in.release()
video_out.release()