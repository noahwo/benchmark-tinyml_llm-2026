import os
import sys
import time
import numpy as np

# Phase 1: Setup

# 1.1 Imports with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
    except Exception as e:
        print("Error: Unable to import TFLite Interpreter. Ensure either 'tflite_runtime' or 'tensorflow' is installed.")
        sys.exit(1)

# Only import cv2 because video/image processing is explicitly required
try:
    import cv2
except Exception as e:
    print("Error: OpenCV (cv2) is required for video I/O. Please install it.")
    sys.exit(1)

# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Utility: deterministic color per class id
def class_color(class_id: int):
    np.random.seed(class_id + 12345)
    color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return color

# 1.3 Load Labels
def load_labels(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"Error: Unable to read label file at {label_file_path}: {e}")
        sys.exit(1)
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
def make_interpreter_with_edgetpu(tflite_model_path):
    last_err = None
    try:
        interpreter = Interpreter(
            model_path=tflite_model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
        return interpreter
    except Exception as e:
        last_err = e
        # Try specific path (common on aarch64)
        try:
            interpreter = Interpreter(
                model_path=tflite_model_path,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
            )
            return interpreter
        except Exception as e2:
            last_err = e2
    print("Error: Failed to load EdgeTPU delegate. Ensure the Coral EdgeTPU runtime is installed and "
          "the device is connected. Details:", last_err)
    sys.exit(1)

interpreter = make_interpreter_with_edgetpu(model_path)
try:
    interpreter.allocate_tensors()
except Exception as e:
    print("Error: Failed to allocate tensors for the TFLite interpreter:", e)
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    print("Error: Model does not have any input tensors.")
    sys.exit(1)

input_shape = input_details[0]['shape']  # Expected shape: [1, height, width, 3]
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Phase 2: Input Acquisition & Preprocessing Loop

# 2.1 Acquire Input Data - open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Unable to open input video file: {input_path}")
    sys.exit(1)

# Prepare output writer with the same dimensions as the input video frames
in_fps = cap.get(cv2.CAP_PROP_FPS)
if in_fps is None or in_fps <= 0 or np.isnan(in_fps):
    in_fps = 30.0  # fallback

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
out_writer = cv2.VideoWriter(output_path, fourcc, in_fps, (frame_width, frame_height))
if not out_writer.isOpened():
    print(f"Error: Unable to open output video writer at: {output_path}")
    cap.release()
    sys.exit(1)

# 2.2 + 2.3 Preprocess frames into model input
def preprocess_frame_bgr_to_model_input(bgr_frame, input_shape, floating_model):
    # Model expects [1, h, w, c]
    _, in_h, in_w, in_c = input_shape
    # Convert BGR to RGB
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h))
    input_data = np.expand_dims(resized, axis=0)

    if floating_model:
        # Normalize to [-1, 1] as per common MobileNet SSD preprocessing
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)
    return input_data

# Helper: parse TFLite detection outputs robustly
def parse_detection_outputs(interpreter, output_details):
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    boxes = classes = scores = count = None
    # First try common order: [boxes, classes, scores, count]
    try:
        boxes = outputs[0][0]
        classes = outputs[1][0]
        scores = outputs[2][0]
        count = int(outputs[3][0]) if outputs[3].size > 0 else len(scores)
        return boxes, classes, scores, count
    except Exception:
        pass
    # Fallback by shape heuristics
    for out in outputs:
        shp = out.shape
        try:
            if len(shp) == 3 and shp[-1] == 4:
                boxes = out[0]
            elif len(shp) == 2:
                arr = out[0]
                # Scores typically float in [0,1]
                if arr.dtype == np.float32 and np.all(arr >= 0) and np.all(arr <= 1.0):
                    scores = arr
                else:
                    classes = arr.astype(np.int32) if arr.dtype != np.int32 else arr
            elif len(shp) == 1 and out.size >= 1:
                # count tensor
                count = int(out[0])
        except Exception:
            continue
    if count is None and scores is not None:
        count = len(scores)
    if boxes is None or classes is None or scores is None or count is None:
        raise RuntimeError("Unable to parse detection outputs from the model.")
    return boxes, classes, scores, count

# 2.4 Loop control variables
total_detections = 0
sum_confidences = 0.0  # For a proxy mAP measure (mean confidence across detections)
frame_index = 0

# Phase 2.4 + 3 + 4 Loop over video frames
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break  # End of video

    frame_index += 1
    orig_h, orig_w = frame_bgr.shape[:2]

    # Preprocess to model input
    input_tensor = preprocess_frame_bgr_to_model_input(frame_bgr, input_shape, floating_model)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling
    # 4.1 Get output tensors
    try:
        boxes, classes, scores, count = parse_detection_outputs(interpreter, output_details)
    except Exception as e:
        print(f"Warning: Failed to parse model outputs at frame {frame_index}: {e}")
        boxes, classes, scores, count = None, None, None, 0

    # 4.2 Interpret Results
    detections_to_draw = []
    if count and boxes is not None and classes is not None and scores is not None:
        for i in range(int(count)):
            score = float(scores[i])
            class_id = int(classes[i]) if i < len(classes) else -1
            label_name = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            # 4.3 Post-processing: thresholding and clipping
            if score >= confidence_threshold:
                # boxes are [ymin, xmin, ymax, xmax] normalized [0,1]
                y_min, x_min, y_max, x_max = boxes[i]
                y_min = max(0.0, min(1.0, float(y_min)))
                x_min = max(0.0, min(1.0, float(x_min)))
                y_max = max(0.0, min(1.0, float(y_max)))
                x_max = max(0.0, min(1.0, float(x_max)))

                # Scale to image coordinates
                x1 = int(x_min * orig_w)
                y1 = int(y_min * orig_h)
                x2 = int(x_max * orig_w)
                y2 = int(y_max * orig_h)

                # Ensure proper box
                x1, x2 = max(0, min(x1, orig_w - 1)), max(0, min(x2, orig_w - 1))
                y1, y2 = max(0, min(y1, orig_h - 1)), max(0, min(y2, orig_h - 1))

                detections_to_draw.append((x1, y1, x2, y2, class_id, label_name, score))
                total_detections += 1
                sum_confidences += score

    # Compute a running "mAP" proxy: mean confidence over all detections so far
    # Note: True mAP requires ground truth annotations. Here we provide a proxy metric due to lack of ground truth.
    map_proxy = (sum_confidences / total_detections) if total_detections > 0 else 0.0

    # 4.4 Handle Output: draw detections and write frame
    # Draw detection rectangles and labels
    for (x1, y1, x2, y2, cid, lname, score) in detections_to_draw:
        color = class_color(cid)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label_text = f"{lname}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # draw filled rectangle for label background
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - baseline - 2)), (x1 + tw + 2, y1), color, thickness=-1)
        cv2.putText(frame_bgr, label_text, (x1 + 1, y1 - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Overlay mAP proxy on frame
    map_text = f"mAP (proxy): {map_proxy:.3f}"
    cv2.putText(frame_bgr, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

    # Write to output video
    out_writer.write(frame_bgr)

# Phase 5: Cleanup
cap.release()
out_writer.release()

# Print final proxy mAP to console
final_map_proxy = (sum_confidences / total_detections) if total_detections > 0 else 0.0
print(f"Processing complete. Output saved at: {output_path}")
print(f"Detections: {total_detections}, mAP (proxy): {final_map_proxy:.4f}")