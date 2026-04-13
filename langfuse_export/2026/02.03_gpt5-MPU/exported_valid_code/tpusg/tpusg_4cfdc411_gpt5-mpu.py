import os
import sys
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tfl_source = "tflite_runtime"
except ImportError:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
        tfl_source = "tensorflow.lite"
    except ImportError as e:
        print("Error: Neither 'tflite_runtime' nor 'tensorflow.lite' could be imported.")
        print("Please install 'python3-tflite-runtime' on the Coral Dev Board, or TensorFlow Lite.")
        sys.exit(1)

import cv2  # Needed for video I/O and drawing

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold_str  = 0.5
try:
    confidence_threshold = float(confidence_threshold_str)
except Exception:
    confidence_threshold = 0.5

# 1.3 Load Labels
def load_labels(labels_file):
    labels = []
    try:
        with open(labels_file, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    except Exception as e:
        print(f"Error loading labels from {labels_file}: {e}")
        labels = []
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
interpreter = None
delegate_error_msgs = []
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e1:
    delegate_error_msgs.append(str(e1))
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e2:
        delegate_error_msgs.append(str(e2))
        print("Failed to load EdgeTPU delegate. Make sure the EdgeTPU runtime is installed.")
        print("Attempted delegates: 'libedgetpu.so.1.0' and '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'.")
        print("Error messages:")
        for i, msg in enumerate(delegate_error_msgs, 1):
            print(f"  {i}. {msg}")
        sys.exit(1)

try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error allocating tensors: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if not input_details:
    print("Error: No input details found in the model.")
    sys.exit(1)

input_shape = input_details[0]['shape']
input_height = int(input_shape[1])
input_width = int(input_shape[2])
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Utility functions for output parsing and drawing
def clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def normalized_to_pixel_coords(box, img_w, img_h):
    # box: [ymin, xmin, ymax, xmax] normalized
    ymin = clip(box[0], 0.0, 1.0)
    xmin = clip(box[1], 0.0, 1.0)
    ymax = clip(box[2], 0.0, 1.0)
    xmax = clip(box[3], 0.0, 1.0)

    x1 = int(xmin * img_w)
    y1 = int(ymin * img_h)
    x2 = int(xmax * img_w)
    y2 = int(ymax * img_h)

    x1 = clip(x1, 0, img_w - 1)
    y1 = clip(y1, 0, img_h - 1)
    x2 = clip(x2, 0, img_w - 1)
    y2 = clip(y2, 0, img_h - 1)
    return x1, y1, x2, y2

def get_detection_tensors(interpreter, output_details):
    # Attempt to robustly extract boxes, classes, scores, and num_detections
    boxes = None
    classes = None
    scores = None
    num_detections = None

    for od in output_details:
        tensor = interpreter.get_tensor(od['index'])
        shape = tensor.shape
        if len(shape) == 3 and shape[-1] == 4:
            boxes = tensor[0]
        elif len(shape) == 2:
            # Could be classes or scores
            if tensor.dtype == np.float32:
                # scores are float32 [0,1]
                if np.max(tensor) <= 1.0 + 1e-6:
                    scores = tensor[0]
                else:
                    # Rare case: classes float (e.g., floats that represent ints)
                    classes = tensor[0].astype(np.int32)
            else:
                classes = tensor[0].astype(np.int32)
        elif len(shape) == 1 and shape[0] == 1:
            # num detections
            num_detections = int(tensor[0])

    # Fallbacks if some are None (common TFLite SSD order)
    if boxes is None or scores is None or classes is None:
        try:
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
        except Exception:
            pass

    if num_detections is None:
        # Fallback to length of scores or boxes
        if scores is not None:
            num_detections = len(scores)
        elif boxes is not None:
            num_detections = boxes.shape[0]
        else:
            num_detections = 0

    return boxes, classes, scores, num_detections

def compute_proxy_map(per_class_conf_scores):
    # Proxy mAP without ground-truth: mean of average confidences per class
    # If no detections at all, return 0.0
    if not per_class_conf_scores:
        return 0.0
    ap_values = []
    for cls_id, confs in per_class_conf_scores.items():
        if confs:
            ap_values.append(float(np.mean(confs)))
    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))

def get_class_color(class_id):
    # Deterministic color for class id
    np.random.seed(class_id + 12345)
    color = tuple(int(c) for c in np.random.randint(64, 256, size=3))
    return color

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Cannot open input video at '{input_path}'")
    sys.exit(1)

orig_fps = cap.get(cv2.CAP_PROP_FPS)
if orig_fps is None or orig_fps <= 0:
    orig_fps = 30.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_width <= 0 or frame_height <= 0:
    print("Error: Invalid video frame dimensions.")
    cap.release()
    sys.exit(1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
try:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
except Exception:
    pass
out_writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (frame_width, frame_height))
if not out_writer.isOpened():
    print(f"Error: Cannot open VideoWriter for '{output_path}'")
    cap.release()
    sys.exit(1)

per_class_conf_scores = {}  # class_id -> list of confidence scores (for proxy mAP)

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # 2.2 Preprocess Data
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(frame_rgb, (input_width, input_height))
        input_data = np.expand_dims(resized_rgb, axis=0)

        # 2.3 Quantization Handling
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = np.asarray(input_data, dtype=input_dtype)

        # Phase 3: Inference
        # 3.1 Set Input Tensor(s)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling Loop
        # 4.1 Get Output Tensor(s)
        boxes, classes, scores, num_detections = get_detection_tensors(interpreter, output_details)

        # 4.2 Interpret Results
        # 4.3 Post-processing (thresholding and clipping)
        if boxes is None or scores is None or classes is None:
            # Nothing to draw; write original frame
            current_map = compute_proxy_map(per_class_conf_scores)
            cv2.putText(frame_bgr, f"mAP: {current_map:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            out_writer.write(frame_bgr)
            continue

        im_h, im_w = frame_bgr.shape[:2]
        valid_count = 0

        for i in range(num_detections):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            class_id = int(classes[i]) if i < len(classes) else -1
            if class_id not in per_class_conf_scores:
                per_class_conf_scores[class_id] = []
            per_class_conf_scores[class_id].append(score)
            valid_count += 1

            box = boxes[i]
            x1, y1, x2, y2 = normalized_to_pixel_coords(box, im_w, im_h)
            color = get_class_color(class_id)

            # Draw rectangle
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            # Label text
            label_str = labels[class_id] if (labels and 0 <= class_id < len(labels)) else f"class_{class_id}"
            caption = f"{label_str}: {score:.2f}"
            # Draw text background to improve readability
            (txt_w, txt_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(y1 - 10, txt_h + 5)
            cv2.rectangle(frame_bgr, (x1, y_text - txt_h - 5), (x1 + txt_w, y_text + baseline - 5), color, -1)
            cv2.putText(frame_bgr, caption, (x1, y_text - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Compute and overlay proxy mAP (mean of per-class average confidences)
        current_map = compute_proxy_map(per_class_conf_scores)
        cv2.putText(frame_bgr, f"mAP: {current_map:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 4.4 Handle Output (write annotated frame to output video)
        out_writer.write(frame_bgr)

        # 4.5 Loop Continuation (handled by reading next frame)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Phase 5: Cleanup
    cap.release()
    out_writer.release()

end_time = time.time()
elapsed = end_time - start_time if end_time > start_time else 0.0
final_map = compute_proxy_map(per_class_conf_scores)
print("Processing complete.")
print(f"Frames processed: {frame_count}")
print(f"Elapsed time (s): {elapsed:.2f}")
if elapsed > 0 and frame_count > 0:
    print(f"Average FPS: {frame_count / elapsed:.2f}")
print(f"Output saved to: {output_path}")
print(f"Calculated mAP (proxy without ground-truth): {final_map:.4f}")