"""
Extracted from: raw_phi4_85a9_tpusg_batch
Entry ID: a2023b4a
Entry Name: a202_tpu_sketch_generator
Session ID: phi4_85a9_tpusg_batch
Timestamp: 2025-07-30T13:49:19.728000+00:00
Tags: benchmark, phi4:latest, tpu_sketch_generator
"""

import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
MODEL_PATH  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_VIDEO_PATH  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
OUTPUT_VIDEO_PATH  = "/home/mendel/tinyml_autopilot/results/sheeps_detections_a202.mp4"
CONFIDENCE_THRESHOLD  = 0.5

# Phase 1: Setup
def load_labels(path):
    with open(path, 'r') as file:
        return [line.strip() for line in file.readlines()]

try:
    labels = load_labels(LABEL_PATH)
except FileNotFoundError:
    print(f"Label file not found at {LABEL_PATH}")
    raise

try:
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except RuntimeError:
    try:
        interpreter = Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        print("Failed to load Edge TPU delegate:", str(e))
        raise

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
video_capture = cv2.VideoCapture(INPUT_VIDEO_PATH)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

floating_model = (input_details[0]['dtype'] == np.float32)
if not floating_model:
    input_scale, input_zero_point = input_details[0]["quantization"]
else:
    input_scale = 1.0
    input_zero_point = 0

# Phase 4: Output Interpretation & Handling Loop
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Preprocess frame for the model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    
    input_data = np.expand_dims(img_resized, axis=0).astype(input_details[0]['dtype'])
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Phase 3: Inference
    start_time = cv2.getTickCount()
    interpreter.invoke()
    inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

    # Phase 4.1: Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    # Phase 4.2: Interpret Results
    for i in range(len(scores)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            
            # Phase 4.3: Post-processing
            label = f"{labels[int(classes[i])]}: {int(scores[i]*100)}%"
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            cv2.putText(frame, label, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

    # Display the resulting frame
    video_writer.write(frame)
    
    # Optional: Show the frame
    # cv2.imshow('Video', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Phase 5: Cleanup
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processed video saved at {OUTPUT_VIDEO_PATH}")