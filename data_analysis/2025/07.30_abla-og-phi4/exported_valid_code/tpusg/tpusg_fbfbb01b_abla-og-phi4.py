import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        raise RuntimeError("Failed to load the model with any available EdgeTPU delegate.") from e

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape and details
input_shape = input_details[0]['shape']
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError("Error opening video file.")

output_video_writer = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for object detection model input
    img_size = (input_shape[1], input_shape[2])
    image_data = cv2.resize(frame, img_size)
    image_data = np.expand_dims(image_data, axis=0)

    if floating_model:
        image_data = (np.float32(image_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Initialize video writer if not already done
    if output_video_writer is None:
        height, width, _ = frame.shape
        output_video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (width, height)
        )

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * frame.shape[1]), int(xmax * frame.shape[1]), int(ymin * frame.shape[0]), int(ymax * frame.shape[0])
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f"{object_name}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    output_video_writer.write(frame)  # Write the frame with bounding boxes to output video

# Phase 5: Cleanup
cap.release()
if output_video_writer:
    output_video_writer.release()
cv2.destroyAllWindows()