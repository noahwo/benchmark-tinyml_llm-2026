import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

# Configuration Parameters (as per your setup)
MODEL_PATH  = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_MAP_PATH  = "models/ssd-mobilenet_v1/labelmap.txt"
IP_CAMERA_STREAM_URL = 'rtsp://<username>:<password>@<ip_address>:554/Streaming/Channels/101'

def load_labels(label_map_path):
    """Loads labels from the file specified in the label map path."""
    with open(label_map_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_frame(frame, input_details):
    """Preprocesses a frame to match model's expected input shape and type."""
    input_shape = input_details[0]['shape']
    # Resize frame
    frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    # Convert frame to float32 and normalize if necessary based on your model's requirement
    frame = np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)
    return frame

def interpret_output(output_data, labels):
    """Interprets the output of the model."""
    # Assuming a classification task; adjust based on your specific use case
    class_id = np.argmax(output_data[0])
    return labels[class_id]

def main():
    # Load labels
    labels = load_labels(LABEL_MAP_PATH)
    
    # Initialize interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Open video stream
    cap = cv2.VideoCapture(IP_CAMERA_STREAM_URL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame, input_details)
        
        # Set tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
        interpreter.invoke()
        
        # Get output and interpret it
        output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        label = interpret_output(output_data, labels)
        
        print(f"Detected: {label}")
        
        # Display the resulting frame with detection label (optional)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()