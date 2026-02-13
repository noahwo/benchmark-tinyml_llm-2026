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

# Get labels from label map file
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Open video files
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the image
    input_shape = input_details[0]['shape']
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_img, axis=0)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    h, w, _ = frame.shape
    
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            # Get bounding box coordinates and draw rectangle on the image
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
            
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            
            # Add label and confidence score
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detections
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()