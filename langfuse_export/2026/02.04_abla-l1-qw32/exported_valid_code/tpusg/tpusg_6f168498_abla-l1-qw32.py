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

# Initialize the interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load labels
labels = read_label_file(label_path)

# Open video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit('ERROR: Unable to open input video.')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Read a new frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image and resize it to fit the model input size
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize((interpreter.get_input_details()[0]['shape'][1], interpreter.get_input_details()[0]['shape'][2]), Image.ANTIALIAS)

    # Run inference
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    objs = detect.get_objects(interpreter, score_threshold=confidence_threshold)

    # Draw bounding boxes and labels on the frame
    for obj in objs:
        bbox = obj.bbox
        label = labels.get(obj.id, 'Unknown')
        score = obj.score

        # Scale the bounding box to the original image size
        xmin = int(bbox.xmin * frame_width)
        ymin = int(bbox.ymin * frame_height)
        xmax = int(bbox.xmax * frame_width)
        ymax = int(bbox.ymax * frame_height)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label_text = f'{label}: {score:.2f}'
        cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()