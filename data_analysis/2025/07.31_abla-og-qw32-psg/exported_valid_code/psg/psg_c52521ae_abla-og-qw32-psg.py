import numpy as np
from tflite_runtime.interpreter import Interpreter

# ### CONFIGURATION PARAMETERS ###
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Confidence threshold for object detection (if applicable)
confidence_threshold  = 0.5

# Phase 1: Setup
# 1.3 Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
import cv2

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError("Cannot open video file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame to match input tensor shape and dtype
    input_shape = input_details[0]['shape']
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb, axis=0).astype(input_details[0]['dtype'])

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # 4.1 Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # 4.2 Interpret Results
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            box = boxes[0][i]
            class_id = int(classes[0][i])
            label = labels[class_id]
            
            # Convert normalized coordinates to image coordinates
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[0][i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 4.3 Handle Output
    out.write(frame)
    
# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()