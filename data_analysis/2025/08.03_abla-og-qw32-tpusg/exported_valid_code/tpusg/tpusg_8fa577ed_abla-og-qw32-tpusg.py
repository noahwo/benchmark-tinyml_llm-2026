import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Phase 1: Setup
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Ensure OpenCV is imported correctly
try:
    import cv2
except ImportError as e:
    print(f"OpenCV could not be imported: {e}")
    raise

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter with EdgeTPU
try:
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
except Exception as e:
    try:
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except Exception as e2:
        print(f"Failed to load EdgeTPU delegate: {e}, {e2}")
        raise

interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = (input_details[0]['dtype'] == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    
    # Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = []
    for i in range(len(output_details)):
        output_data.append(interpreter.get_tensor(output_details[i]['index']))

    boxes, class_ids, scores, num_detections = output_data

    num_detections = int(num_detections[0])
    for i in range(num_detections):
        score = float(scores[0][i])
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = map(int, (boxes[0][i] * np.array([height, width, height, width])))
            class_id = int(class_ids[0][i])
            label = labels[class_id]
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()