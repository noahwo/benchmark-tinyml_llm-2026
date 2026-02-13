import cv2
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Configuration parameters
model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
input_path = '/home/mendel/tinyml_autopilot/data/sheeps.mp4'
output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load the TFLite model and labels
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
labels = read_label_file(label_path)

# Open the input video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit('ERROR: Unable to open video source.')

# Get video properties to initialize output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a format suitable for the model
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize((300, 300), Image.ANTIALIAS)

    # Run inference
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    objs = detect.get_objects(interpreter, confidence_threshold)

    # Draw detection results on the frame
    for obj in objs:
        bbox = obj.bbox
        label = labels.get(obj.id, obj.id)
        score = int(obj.score * 100)
        display_str = f'{label}: {score}%'
        
        # Scale bounding box coordinates to original image size
        ymin, xmin, ymax, xmax = bbox
        ymin, xmin, ymax, xmax = (ymin * frame_height, xmin * frame_width,
                                  ymax * frame_height, xmax * frame_width)
        
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, display_str, (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write the frame with detection results to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()