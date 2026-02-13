# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np

# Phase 1.2: Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1.3: Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Phase 1.4: Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Phase 1.5: Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input tensor shape and type from details
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 2.2: Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # Get all output details
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence score of detected objects

    # Phase 4.2: Interpret Results
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Filter out weak detections by a threshold
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            label = labels[int(classes[i])]
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label_text = f"{label}: {scores[i]:.2f}"
            cv2.putText(frame, label_text, (int(left), int(top) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Phase 4.3: Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_path)