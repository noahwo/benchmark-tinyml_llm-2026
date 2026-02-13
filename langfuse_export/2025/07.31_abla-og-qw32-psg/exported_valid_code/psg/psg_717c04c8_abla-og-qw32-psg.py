import numpy as np
from tflite_runtime.interpreter import Interpreter
import cv2

### CONFIGURATION PARAMETERS ###
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH  = "data/object_detection/sheeps.mp4"
OUTPUT_PATH  = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD  = 0.5

### Phase 1: Setup ###
# 1.1 Imports: Already done at the top of the script.
# 1.2 Paths/Parameters: Defined above.
# 1.3 Load Labels (Conditional)
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1 Acquire Input Data (Video Capture)
cap = cv2.VideoCapture(INPUT_PATH)

# Check if the video was opened successfully
if not cap.isOpened():
    raise IOError("Error opening video stream or file")

# Get video properties to save output video with same properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    # 2.2 Preprocess Data
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    # Phase 3: Inference
    # 3.1 Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
    
    # 3.2 Run Inference
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    # 4.1 Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]   # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]     # Confidence score of detections
    
    # 4.2 Interpret Results
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        if score > CONFIDENCE_THRESHOLD:
            # Convert normalized coordinates to pixel values
            ymin = int(box[0] * height)
            xmin = int(box[1] * width)
            ymax = int(box[2] * height)
            xmax = int(box[3] * width)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f'{labels[int(cls)]}: {score:.2f}'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 4.3 Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()