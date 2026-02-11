### CONFIGURATION PARAMETERS ###
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

### Phase 1: Setup ###
# 1.1. Imports
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np

# 1.2. Paths/Parameters (already defined in CONFIGURATION PARAMETERS section)

# 1.3. Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4. Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5. Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ensure the model expects a single image input and outputs detection boxes, classes, scores, and num_detections
input_shape = input_details[0]['shape']
floating_model = (input_details[0]['dtype'] == np.float32)

### Phase 2: Input Acquisition & Preprocessing Loop ###
# 2.1. Acquire Input Data
cap = cv2.VideoCapture(input_path)  # Assuming the input path is a video file

# Define the codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image to fit the model's expected input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # 2.3. Loop Control: Continue processing each frame in the video

    ### Phase 3: Inference ###
    # 3.1. Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 3.2. Run Inference
    interpreter.invoke()

    ### Phase 4: Output Interpretation & Handling Loop ###
    # 4.1. Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # 4.2. Interpret Results
    height, width, _ = frame.shape

    for i in range(num_detections):
        if scores[i] > 0.5:  # Only consider detections with a score higher than 0.5
            ymin = int(max(1, (boxes[i][0] * height)))
            xmin = int(max(1, (boxes[i][1] * width)))
            ymax = int(min(height, (boxes[i][2] * height)))
            xmax = int(min(width, (boxes[i][3] * width)))

            label = labels[int(classes[i])]
            score = scores[i]
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 4.3. Handle Output
    out.write(frame)  # Write the processed frame to the output video

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()