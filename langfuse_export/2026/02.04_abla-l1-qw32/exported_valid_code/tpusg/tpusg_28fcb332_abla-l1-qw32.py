import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the interpreter with Edge TPU delegate
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the video file for reading
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Video writer setup
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = frame_rgb.reshape(input_shape)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Extract results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw detections on the frame
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]

            # Convert box coordinates to pixel values
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)

            # Draw rectangle and label on frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),
                          (0, 255, 0), thickness=2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()