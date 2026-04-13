import os
import sys
import time
import numpy as np
import cv2

# ============================
# Phase 1: Setup
# ============================

# 1.1 Imports with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_backend = "tflite_runtime"
except Exception as e_rt:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        tflite_backend = "tensorflow.lite"
    except Exception as e_tf:
        print("ERROR: Failed to import TFLite Interpreter. Tried tflite_runtime and tensorflow.lite.")
        print(f"tflite_runtime error: {e_rt}")
        print(f"tensorflow.lite error: {e_tf}")
        sys.exit(1)

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3 Load Labels
def load_labelmap(path):
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"Warning: Unable to load labels from {path}: {e}")
    return labels

labels_list = load_labelmap(label_path)

# 1.4 Load Interpreter with EdgeTPU
def make_interpreter_with_edgetpu(model_file):
    last_error_1 = None
    last_error_2 = None
    try:
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
        return interpreter, 'libedgetpu.so.1.0'
    except Exception as e1:
        last_error_1 = e1
        try:
            interpreter = Interpreter(
                model_path=model_file,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
            )
            return interpreter, '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'
        except Exception as e2:
            last_error_2 = e2
            print("ERROR: Failed to load EdgeTPU delegate. Tried:")
            print(" - 'libedgetpu.so.1.0'")
            print(" - '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'")
            print(f"First attempt error: {last_error_1}")
            print(f"Second attempt error: {last_error_2}")
            print("Please ensure the EdgeTPU runtime is installed and the device is connected.")
            raise

try:
    interpreter, delegate_lib = make_interpreter_with_edgetpu(model_path)
except Exception:
    sys.exit(1)

try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"ERROR: Failed to allocate tensors: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if not input_details:
    print("ERROR: No input details found in the model.")
    sys.exit(1)

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Log basic info
print("=== Configuration ===")
print(f"Backend: {tflite_backend}, Delegate: {delegate_lib}")
print(f"Model: {model_path}")
print(f"Labels: {label_path} ({len(labels_list)} classes loaded)")
print(f"Input video: {input_path}")
print(f"Output video: {output_path}")
print(f"Confidence threshold: {confidence_threshold}")
print(f"Model input shape: {input_shape}, dtype: {input_dtype}")

# ============================
# Helpers for output parsing
# ============================

def get_output_tensors(interp, out_details):
    outputs = []
    for od in out_details:
        outputs.append(interp.get_tensor(od['index']))
    return outputs

def parse_detection_outputs(raw_outputs):
    """
    Attempts to identify boxes, classes, scores, and num_detections from TFLite Detection PostProcess outputs.
    Expected typical shapes:
      - boxes: (1, N, 4)
      - classes: (1, N)
      - scores: (1, N)
      - num_detections: (1,)
    """
    boxes = None
    classes = None
    scores = None
    num_detections = None

    # Identify boxes by last dim == 4
    for arr in raw_outputs:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr

    # Remaining arrays of shape (1, N) or (1,)
    candidates_1xn = [arr for arr in raw_outputs if arr is not boxes]

    # Identify num_detections as shape (1,) or values small integer
    for arr in candidates_1xn:
        if arr.ndim == 1 and arr.shape[0] == 1:
            num_detections = arr
            break

    # Identify classes and scores among (1, N)
    one_by_n = [arr for arr in candidates_1xn if arr.ndim == 2]
    # Heuristics: scores are in [0, 1], classes are typically small integer floats
    for arr in one_by_n:
        flat = arr.ravel()
        if flat.size > 0 and np.all((flat >= 0.0) & (flat <= 1.0)):
            scores = arr
            break

    # The remaining (1, N) should be classes
    for arr in one_by_n:
        if scores is not None and arr is scores:
            continue
        classes = arr
        break

    # Squeeze to (N, ...) shapes
    if boxes is not None:
        boxes = np.squeeze(boxes)
    if classes is not None:
        classes = np.squeeze(classes).astype(np.int32)
    if scores is not None:
        scores = np.squeeze(scores)
    if num_detections is not None:
        # Some models output float32 count
        num_detections = int(np.squeeze(num_detections).astype(np.int32))

    # Fallbacks if num_detections missing
    if num_detections is None and scores is not None:
        num_detections = scores.shape[0]
    if num_detections is None and boxes is not None:
        num_detections = boxes.shape[0]

    return boxes, classes, scores, num_detections

def draw_detections_on_frame(frame_bgr, detections, labels, conf_thresh):
    """
    detections: list of dict with keys: 'bbox' (xmin, ymin, xmax, ymax), 'score', 'class_id'
    """
    for det in detections:
        xmin, ymin, xmax, ymax = det['bbox']
        score = det['score']
        class_id = det['class_id']
        if labels and 0 <= class_id < len(labels):
            label_text = labels[class_id]
        else:
            label_text = f"id:{class_id}"
        caption = f"{label_text} {score:.2f}"

        # Draw rectangle
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Draw label background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (xmin, max(0, ymin - th - baseline)),
                      (xmin + tw, ymin), (0, 255, 0), thickness=-1)
        # Put label text
        cv2.putText(frame_bgr, caption, (xmin, max(0, ymin - baseline)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

# ============================
# Phase 2: Input Acquisition & Preprocessing Loop
# ============================

# 2.1 Acquire Input Data - Open video file
video_cap = cv2.VideoCapture(input_path)
if not video_cap.isOpened():
    print(f"ERROR: Cannot open video file: {input_path}")
    sys.exit(1)

orig_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0  # fallback

# Prepare video writer
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
if not video_writer.isOpened():
    print(f"ERROR: Cannot open video writer for: {output_path}")
    video_cap.release()
    sys.exit(1)

# Determine model input size
if len(input_shape) == 4:
    _, in_h, in_w, in_c = input_shape
else:
    print(f"ERROR: Unexpected input tensor shape: {input_shape}")
    video_cap.release()
    video_writer.release()
    sys.exit(1)

# Variables for simple performance reporting
frame_count = 0
total_inference_time = 0.0

print("Starting inference on video...")

# ============================
# Processing Loop
# ============================
while True:
    ret, frame_bgr = video_cap.read()
    if not ret:
        break
    frame_count += 1

    # 2.2 Preprocess Data
    # Convert BGR to RGB, resize to model input size
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    input_data = np.expand_dims(resized_rgb, axis=0)
    # 2.3 Quantization Handling
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.asarray(input_data, dtype=input_dtype)

    # ============================
    # Phase 3: Inference
    # ============================
    try:
        interpreter.set_tensor(input_index, input_data)
    except Exception as e:
        print(f"ERROR: Failed to set input tensor: {e}")
        break

    start_invoke = time.time()
    try:
        interpreter.invoke()
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        break
    end_invoke = time.time()
    inf_time_ms = (end_invoke - start_invoke) * 1000.0
    total_inference_time += (end_invoke - start_invoke)

    # ============================
    # Phase 4: Output Interpretation & Handling
    # ============================

    # 4.1 Get Output Tensor(s)
    try:
        raw_outputs = get_output_tensors(interpreter, output_details)
    except Exception as e:
        print(f"ERROR: Failed to get output tensors: {e}")
        break

    # 4.2 Interpret Results
    boxes, classes, scores, num_detections = parse_detection_outputs(raw_outputs)

    detections = []
    if boxes is not None and classes is not None and scores is not None and num_detections is not None:
        # 4.3 Post-processing: thresholding, scaling, clipping
        n = int(num_detections)
        for i in range(n):
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < confidence_threshold:
                continue

            # boxes in format [ymin, xmin, ymax, xmax] normalized [0,1]
            if i < len(boxes):
                y_min = float(boxes[i][0])
                x_min = float(boxes[i][1])
                y_max = float(boxes[i][2])
                x_max = float(boxes[i][3])
            else:
                continue

            # Clip to [0,1]
            y_min = max(0.0, min(1.0, y_min))
            x_min = max(0.0, min(1.0, x_min))
            y_max = max(0.0, min(1.0, y_max))
            x_max = max(0.0, min(1.0, x_max))

            # Scale to image size
            xmin_px = int(x_min * orig_width)
            ymin_px = int(y_min * orig_height)
            xmax_px = int(x_max * orig_width)
            ymax_px = int(y_max * orig_height)

            # Ensure proper ordering and clip to frame bounds
            xmin_px = max(0, min(orig_width - 1, xmin_px))
            xmax_px = max(0, min(orig_width - 1, xmax_px))
            ymin_px = max(0, min(orig_height - 1, ymin_px))
            ymax_px = max(0, min(orig_height - 1, ymax_px))
            if xmax_px <= xmin_px or ymax_px <= ymin_px:
                continue

            class_id = int(classes[i]) if i < len(classes) else -1
            detections.append({
                'bbox': (xmin_px, ymin_px, xmax_px, ymax_px),
                'score': score,
                'class_id': class_id
            })

    # Draw detections on the frame
    draw_detections_on_frame(frame_bgr, detections, labels_list, confidence_threshold)

    # Put inference time and note about mAP on the frame
    # mAP requires ground truth annotations; since none provided, we mark it as N/A
    info_text = f"Inference: {inf_time_ms:.1f} ms | mAP: N/A (no ground truth)"
    cv2.putText(frame_bgr, info_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(frame_bgr, info_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # 4.4 Handle Output - write frame to output video
    video_writer.write(frame_bgr)

    # 4.5 Loop Continuation - continues until video ends

# ============================
# Phase 5: Cleanup
# ============================

video_cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Print summary
if frame_count > 0:
    avg_inf_ms = (total_inference_time / frame_count) * 1000.0
    print(f"Processed {frame_count} frames.")
    print(f"Average inference time: {avg_inf_ms:.2f} ms per frame.")
else:
    print("No frames were processed.")

print("mAP: N/A (mean Average Precision requires ground truth annotations, which were not provided).")
print(f"Output saved to: {output_path}")