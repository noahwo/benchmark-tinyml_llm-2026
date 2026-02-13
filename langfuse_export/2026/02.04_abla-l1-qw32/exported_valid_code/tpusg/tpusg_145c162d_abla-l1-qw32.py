import cv2
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
def load_labels(filename):
    with open(filename, 'r', encoding='utf8') as f:
        return {index: line.strip() for index, line in enumerate(f.readlines())}

labels = load_labels(label_path)

# Initialize TFLite interpreter with Edge TPU delegate
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check the type of the input tensor
floating_model = (input_details[0]['dtype'] == np.float32)

# Read video file
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    input_data = np.expand_dims(img, axis=0)

    # Normalize pixel values if using a floating model (note: normalization is not always necessary and depends on the model)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Postprocess the results
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            # Draw bounding box
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

            # Draw label and confidence score
            class_id = int(classes[i])
            label = f'{labels.get(class_id, "Unknown")}: {int(scores[i]*100)}%'
            position = (int(left), int(top) - 10)
            cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # Write the frame with detection to the output file
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()