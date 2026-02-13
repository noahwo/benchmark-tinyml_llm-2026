import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

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
except RuntimeError as e:
    print(f"Failed to load EdgeTPU delegate with 'libedgetpu.so.1.0': {e}")
    try:
        interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
    except RuntimeError as e_inner:
        print(f"Failed to load EdgeTPU delegate with '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0': {e_inner}")
        raise

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_frame(frame):
    """Resize and pad image to expected size."""
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    
    # Resize image with unchanged aspect ratio using padding
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(scale * w), int(scale * h)
    frame_resized = cv2.resize(frame, (nw, nh))

    top_pad = (height - nh) // 2
    bottom_pad = height - nh - top_pad
    left_pad = (width - nw) // 2
    right_pad = width - nw - left_pad
    
    frame_padded = cv2.copyMakeBorder(frame_resized,
                                      top_pad, bottom_pad, 
                                      left_pad, right_pad, 
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return np.expand_dims(frame_padded, axis=0).astype(np.uint8)

def postprocess_output(output_data):
    """Process the detection output to filter by confidence threshold and prepare for drawing."""
    boxes = output_data[0][0]  # Bounding box coordinates
    class_ids = output_data[1][0].astype(np.int32)  # Class indices
    scores = output_data[2][0]  # Confidence scores

    results = []
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * input_width, xmax * input_width,
                                          ymin * input_height, ymax * input_height)
            results.append((int(left), int(top), int(right), int(bottom), scores[i], class_ids[i]))
    return results

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError("Cannot open video file")

with open(output_path, 'w') as output_file:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        input_width, input_height = input_details[0]['shape'][2], input_details[0]['shape'][1]
        preprocessed_frame = preprocess_frame(frame)

        # Phase 3: Inference
        interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
        interpreter.invoke()

        output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        detections = postprocess_output(output_data)

        # Draw bounding boxes and labels on the frame
        for (left, top, right, bottom, score, class_id) in detections:
            label = f"{labels[class_id]}: {score:.2f}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', frame)

        # Write output to file if needed
        # Here we assume output_file is for demonstration purposes; adapt as necessary.
        output_file.write(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}, Detections: {detections}\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Phase 5: Cleanup
cap.release()
cv2.destroyAllWindows()

print("Processing completed and results saved.")