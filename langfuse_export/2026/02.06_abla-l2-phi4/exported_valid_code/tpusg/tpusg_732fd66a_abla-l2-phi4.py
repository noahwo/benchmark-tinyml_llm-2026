import os
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

def main():
    # Configuration parameters
    model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
    label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
    input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
    output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
    confidence_threshold = 0.5

    # Load labels
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize the TFLite interpreter with EdgeTPU support
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get video capture and writer
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing: Resize and convert to UINT8
        input_shape = input_details[0]['shape']
        image_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
        image_np = np.expand_dims(image_resized, axis=0).astype(np.uint8)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], image_np)
        interpreter.invoke()

        # Output handling: Get detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence of detected objects

        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * width, xmax * width,
                                              ymin * height, ymax * height)
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = f"{object_name}: {int(scores[i]*100)}%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                top = max(top, labelSize[1])
                cv2.rectangle(frame, (int(left), int(top) - round(1.5*labelSize[1])),
                              (int(left)+round(1.5*labelSize[0]), int(top+baseLine)),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (int(left), int(top) + baseLine),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Write the frame with detections
        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()