import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration Parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# Open video files
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, (width, height))
    input_data = np.expand_dims(resized_img, axis=0).astype(np.uint8)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence

    # Loop over all detections and draw detection boxes with labels
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            
            # Draw rectangle and label
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            object_name = labels[int(classes[i])] if int(classes[i]) < len(labels) else 'N/A'
            label = f"{object_name}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 1)

    # Write the frame with boxes
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Object detection completed and video saved at:", output_path)