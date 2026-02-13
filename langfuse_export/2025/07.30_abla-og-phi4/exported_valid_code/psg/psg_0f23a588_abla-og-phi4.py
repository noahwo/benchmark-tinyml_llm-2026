import cv2  # OpenCV library for video processing
from tflite_runtime.interpreter import Interpreter  # TensorFlow Lite interpreter

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
video_input_path = 'path_to_your_video.mp4'  # Replace with your actual video input path
label_map_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Load labels if provided and relevant
labels = []
with open(label_map_path, 'r') as file:
    for line in file.readlines():
        labels.append(line.strip())

# Load the interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details  = "data/object_detection/sheeps.mp4"
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(video_input_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert the image to match input requirements
    input_shape = input_details[0]['shape']
    input_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = input_frame.astype('float32')

    # Expand dimensions to match model's expected input
    input_data = np.expand_dims(input_frame, axis=0)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence score

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Threshold can be adjusted based on requirements
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)

            object_name = labels[int(classes[i])] if classes and len(labels) > int(classes[i]) else 'N/A'
            label = f"{object_name}: {int(scores[i]*100)}%"
            cv2.putText(frame, label, (int(left), int(top)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 5: Cleanup
cap.release()
cv2.destroyAllWindows()