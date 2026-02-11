import cv2
import numpy as np
import tensorflow as tf

def load_labels(label_path):
    with open(label_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

def detect_objects(frame, interpreter, input_details, output_details, input_size, threshold=0.5):
    # Preprocess the image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_size, input_size))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Filter out low confidence detections and draw boxes
    h, w, _ = frame.shape
    results = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            results.append((classes[i], scores[i], (left, top, right, bottom)))

    return results

def main():
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
    input_size = input_details[0]['shape'][1]

    # Load labels
    labels = load_labels(label_path)

    # Open video file or capture device
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        detections = detect_objects(frame, interpreter, input_details, output_details, input_size, confidence_threshold)

        # Draw bounding boxes and labels on the frame
        for class_id, score, box in detections:
            label = f"{labels[int(class_id)]}: {score:.2f}"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with detections
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()