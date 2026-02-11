import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# CONFIGURATION PARAMETERS
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except ValueError:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape requirements
required_height = input_details[0]['shape'][1]
required_width = input_details[0]['shape'][2]

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare output video writer
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

floating_model = (input_details[0]['dtype'] == np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the input image to required dimensions
    resized_frame = cv2.resize(frame, (required_width, required_height))
    
    # Preprocess the input
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Phase 3: Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    for detection in output_data:
        scores = detection[5:]
        
        if len(scores) == 0:
            continue  # Skip this detection as there are no class scores

        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > confidence_threshold:
            # Bounding box information
            ymin, xmin, ymax, xmax = detection[0:4]
            
            # Scale bounding box to original image dimensions
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            text = f'{labels[class_id]}: {confidence:.2f}'
            cv2.putText(frame, text, (int(left), int(top) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Handle Output
    out.write(frame)

cap.release()
out.release()

# Phase 5: Cleanup
cv2.destroyAllWindows()