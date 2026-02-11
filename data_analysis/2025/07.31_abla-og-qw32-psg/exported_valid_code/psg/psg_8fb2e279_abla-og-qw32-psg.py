# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# Configuration Parameters
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

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 2.2: Preprocess Data
    input_shape = input_details[0]['shape']
    if len(input_shape) == 4 and input_shape[1] > 1:
        target_height = input_shape[1]
        target_width = input_shape[2]
        resized_frame = cv2.resize(frame, (target_width, target_height))
        input_data = np.expand_dims(resized_frame, axis=0)
    else:
        raise ValueError("Input tensor shape is not compatible with expected format.")

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Example interpretation (assuming single class classification or detection)
    # Here we assume the model outputs a probability distribution for each label
    if len(output_data.shape) == 2:
        scores = output_data[0]
        top_k_indices = np.argsort(scores)[::-1][:5]  # Get indices of top 5 predictions

        for i in range(5):
            index = top_k_indices[i]
            score = scores[index]
            label = labels[index]
            text = f"{label}: {score:.2f}"
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the frame with predictions to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")