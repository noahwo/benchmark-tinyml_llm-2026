import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import ImageDraw, ImageFont

# Configuration parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read video file
cap = cv2.VideoCapture(input_path)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def draw_boxes_on_frame(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            label = labels[int(classes[i])]
            score = scores[i]

            # Convert normalized coordinates to absolute coordinates
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)

            # Draw rectangle and text on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label_text, (int(left), int(top) - 10),
                        font, 0.9, (0, 255, 0), 2)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(image_rgb, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.uint8))
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw bounding boxes on the frame
    draw_boxes_on_frame(frame, boxes, classes, scores)

    # Write the frame with detection to the output video file
    out.write(frame)

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_path}")