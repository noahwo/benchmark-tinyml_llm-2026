import cv2
import numpy as np
import tensorflow as tf

# Configuration parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    interpreter.set_tensor(input_details[0]['index'], image)

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    return interpreter.get_tensor(output_details[index]['index'])

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess the frame
    input_size = input_details[0]['shape'][1:3]
    image_resized = cv2.resize(frame, (input_size[1], input_size[0]))
    input_data = np.expand_dims(image_resized, axis=0)

    # Set the tensor to point to the input data to be inferred
    set_input_tensor(interpreter, input_data)

    # Run inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3)[0])

    # Loop over all detections and draw detection box if confidence is above threshold
    for i in range(count):
        if scores[0][i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0][i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)

            # Draw a rectangle around the object
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[0][i])]
            label = f'{object_name}: {int(scores[0][i] * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - round(1.5 * label_size[1]))),
                          (int(left + round(1.5 * label_size[0])), int(top + base_line)),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()