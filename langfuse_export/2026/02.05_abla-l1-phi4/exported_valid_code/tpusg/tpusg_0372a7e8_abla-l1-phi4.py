import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter with EdgeTPU delegate
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[
        tflite.load_delegate('libedgetpu.so.1')
    ]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input size
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# Open video file
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    img_resized = cv2.resize(frame, (input_width, input_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_data = np.expand_dims(pil_img, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects

    # Draw bounding boxes and labels on the frame
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * cap.get(3), xmax * cap.get(3),
                                          ymin * cap.get(4), ymax * cap.get(4))
            
            # Draw rectangle and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {scores[i]:.2f}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - label_size[1])), 
                          (int(left + label_size[0]), int(top + base_line)), 
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()