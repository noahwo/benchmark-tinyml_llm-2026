from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

### CONFIGURATION PARAMETERS ###
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"

### Phase 1: Setup ###
# 1.1. Imports are already done above.
# 1.2. Paths/Parameters are defined above.

# 1.3. Load Labels (Conditional)
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4. Load Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 1.5. Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

### Phase 2: Input Acquisition & Preprocessing Loop ###
# 2.1. Acquire Input Data (Reading a video file)
cap = cv2.VideoCapture(INPUT_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(image_rgb, axis=0)

    # 2.2. Preprocess Data
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    ### Phase 3: Inference ###
    # 3.1. Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 3.2. Run Inference
    interpreter.invoke()

    ### Phase 4: Output Interpretation & Handling Loop ###
    # 4.1. Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 4.2. Interpret Results (Assuming detection model, adjust as necessary)
    detections = np.squeeze(output_data)

    for i in range(detections.shape[0]):
        if detections[i][2] > 0.5:  # Confidence threshold
            ymin = int(max(1, (detections[i][0] * frame_height)))
            xmin = int(max(1, (detections[i][1] * frame_width)))
            ymax = int(min(frame_height, (detections[i][2] * frame_height)))
            xmax = int(min(frame_width, (detections[i][3] * frame_width)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{labels[int(detections[i][1])]}: {detections[i][2]:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 4.3. Handle Output
    out.write(frame)

### Phase 5: Cleanup ###
# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()