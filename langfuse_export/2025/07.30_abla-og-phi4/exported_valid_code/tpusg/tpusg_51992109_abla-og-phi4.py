import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate
import numpy as np

# Configuration Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    print(f"Failed to load EdgeTPU delegate from 'libedgetpu.so.1.0': {e}")
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e:
        raise RuntimeError(f"Failed to load EdgeTPU delegate from '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0': {e}")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

def preprocess_frame(frame):
    # Resize and pad frame to model input size
    input_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])
    frame_resized = cv2.resize(frame, input_size)
    
    # Check if the model expects float inputs
    floating_model = (input_details[0]['dtype'] == np.float32)

    # Preprocess input for model
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data

def postprocess_output(output_data, frame_shape):
    # Parse detection results
    boxes = output_data[0][0]  # Bounding box coordinates
    classes = output_data[1][0].astype(np.int32)  # Class indices
    scores = output_data[2][0]  # Confidence scores

    height, width, _ = frame_shape
    results = []

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            results.append((labels[classes[i]], scores[i], (xmin, ymin, xmax, ymax)))

    return results

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_frame(frame)
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    detections = postprocess_output([boxes, classes, scores], frame.shape)

    for label, score, box in detections:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()