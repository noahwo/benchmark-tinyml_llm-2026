import cv2
import numpy as np
import tensorflow as tf

# Configuration parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file or stream
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

# Load label map
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = input_details[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_boxes(interpreter):
    """Returns list of detection boxes with confidence > threshold"""
    output_data = [interpreter.get_tensor(output['index']) for output in output_details]

    # Extract boxes, classes, and scores
    boxes = output_data[0][0]  # Bounding box coordinates of detected objects
    classes = output_data[1][0].astype(np.int64)  # Class index of detected objects
    scores = output_data[2][0]  # Confidence of detected objects

    results = []
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            results.append((int(left), int(right), int(top), int(bottom), scores[i], classes[i]))
    return results

def draw_boxes(frame, boxes):
    """Draws bounding box with label and confidence score on the image"""
    for box in boxes:
        left, right, top, bottom, score, class_id = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"{labels[class_id]}: {score:.2f}"
        cv2.putText(frame, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (300, 300))
    input_image = np.expand_dims(input_image, axis=0)

    # Set the input tensor
    set_input_tensor(interpreter, input_image)

    # Run inference
    interpreter.invoke()

    # Get detection results
    boxes = get_output_boxes(interpreter)
    draw_boxes(frame, boxes)

    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()