import os
import sys
import time
import numpy as np

# Phase 1: Setup

# 1.1 Imports: TFLite Interpreter and EdgeTPU delegate with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
    except Exception as e:
        print("ERROR: Unable to import TFLite Interpreter. Ensure tflite_runtime or tensorflow is installed.")
        sys.exit(1)

# Import cv2 only because image/video processing is explicitly required
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is required for video processing but could not be imported.")
    sys.exit(1)

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold_str  = 0.5
try:
    confidence_threshold = float(confidence_threshold_str)
except ValueError:
    print("ERROR: confidence_threshold is not a float. Check configuration.")
    sys.exit(1)

# 1.3 Load Labels (if provided and relevant)
labels = []
if label_path and os.path.isfile(label_path):
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"WARNING: Failed to read label file at {label_path}. Proceeding without labels. Error: {e}")
else:
    if label_path:
        print(f"WARNING: Label file not found at {label_path}. Proceeding without labels.")

# 1.4 Load Interpreter with EdgeTPU delegate
interpreter = None
delegate_errors = []
try:
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e1:
    delegate_errors.append(("libedgetpu.so.1.0", str(e1)))
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e2:
        delegate_errors.append(("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0", str(e2)))
        print("ERROR: Failed to load EdgeTPU delegate with both default and fallback paths.")
        for lib, err in delegate_errors:
            print(f" - Attempted delegate '{lib}' failed with error: {err}")
        print("Ensure the EdgeTPU runtime is installed and the correct shared library is available.")
        sys.exit(1)

# Allocate tensors
try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"ERROR: Failed to allocate tensors: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    print("ERROR: Model does not have any input tensors.")
    sys.exit(1)

input_shape = input_details[0]['shape']
input_height = int(input_shape[1])
input_width = int(input_shape[2])
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Utility functions
def preprocess_frame_bgr_to_model_input(bgr_frame, target_width, target_height, floating):
    # Convert BGR to RGB, resize, and prepare tensor
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_width, target_height))
    input_data = np.expand_dims(resized, axis=0)
    if floating:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)
    return input_data

def safe_label(class_id):
    if labels and 0 <= class_id < len(labels):
        return labels[class_id]
    return f"id_{class_id}"

def clip(value, min_v, max_v):
    return max(min_v, min(value, max_v))

# Phase 2: Input Acquisition & Preprocessing Loop

# 2.1 Acquire Input Data - open video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"ERROR: Unable to open input video file: {input_path}")
    sys.exit(1)

# Retrieve video properties
input_fps = cap.get(cv2.CAP_PROP_FPS)
if input_fps <= 0 or np.isnan(input_fps):
    input_fps = 30.0  # fallback
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
if not writer.isOpened():
    # Try fallback codecs
    for codec in ['avc1', 'XVID', 'MJPG']:
        fourcc_alt = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc_alt, input_fps, (frame_width, frame_height))
        if writer.isOpened():
            break
if not writer.isOpened():
    print(f"ERROR: Unable to open VideoWriter for output file: {output_path}")
    cap.release()
    sys.exit(1)

# Tracking for mAP proxy computation
# We approximate AP per class by average of detection scores above threshold; mAP is mean across classes observed
class_scores = {}  # class_id -> list of scores
frame_count = 0
total_inference_time_ms = 0.0

# 2.4 Loop Control - process entire video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 2.2 Preprocess Data
    input_data = preprocess_frame_bgr_to_model_input(frame, input_width, input_height, floating_model)

    # Phase 3: Inference
    # 3.1 Set Input Tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # 3.2 Run Inference
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    infer_ms = (t1 - t0) * 1000.0
    total_inference_time_ms += infer_ms

    # Phase 4: Output Interpretation & Handling Loop

    # 4.1 Get Output Tensors
    # Typical SSD output tensors: boxes [1,N,4], classes [1,N], scores [1,N], count [1]
    raw_outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    boxes = None
    classes = None
    scores = None
    count = None

    for out in raw_outputs:
        arr = np.squeeze(out)
        if arr.ndim == 2 and arr.shape[1] == 4:
            # boxes
            boxes = arr
        elif arr.ndim == 1:
            # count or 1D of N (classes or scores)
            if arr.shape[0] == 1:
                count = int(arr[0])
            else:
                # Need to distinguish classes vs scores; try by value range
                if np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
                    scores = arr
                else:
                    classes = arr.astype(np.int32)
        elif arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 4:
            boxes = arr[0]
        elif arr.ndim == 2:
            # Could be classes or scores
            if np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
                scores = arr[0] if arr.shape[0] == 1 else arr
            else:
                classes = arr[0].astype(np.int32) if arr.shape[0] == 1 else arr.astype(np.int32)

    # Fallbacks if any is still None due to variant output ordering
    if boxes is None:
        # Attempt by shape from output_details directly
        for od in output_details:
            o = interpreter.get_tensor(od['index'])
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes = o[0]
                break
    if count is None:
        # Estimate count by available size
        if scores is not None:
            count = int(len(scores))
        elif classes is not None:
            count = int(len(classes))
        elif boxes is not None:
            count = int(boxes.shape[0])
        else:
            count = 0

    if scores is None or classes is None or boxes is None:
        # If outputs are still not identified, skip this frame gracefully
        overlay = frame.copy()
        cv2.putText(overlay, "Detection output parsing failed.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(overlay)
        continue

    # Ensure sizes match
    N = min(count, len(scores), len(classes), boxes.shape[0])

    # 4.2 Interpret Results
    detections = []
    for i in range(N):
        score = float(scores[i])
        cls_id = int(classes[i])
        # 4.3 Post-processing: confidence thresholding
        if score < confidence_threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        # Scale from normalized coordinates [0,1] to pixel coordinates
        x_min_abs = int(clip(x_min * frame_width, 0, frame_width - 1))
        y_min_abs = int(clip(y_min * frame_height, 0, frame_height - 1))
        x_max_abs = int(clip(x_max * frame_width, 0, frame_width - 1))
        y_max_abs = int(clip(y_max * frame_height, 0, frame_height - 1))

        # Ensure box has positive area after clipping
        if x_max_abs <= x_min_abs or y_max_abs <= y_min_abs:
            continue

        label_text = safe_label(cls_id)
        detections.append({
            'bbox': (x_min_abs, y_min_abs, x_max_abs, y_max_abs),
            'score': score,
            'class_id': cls_id,
            'label': label_text
        })

        # Accumulate scores for mAP proxy computation
        if cls_id not in class_scores:
            class_scores[cls_id] = []
        class_scores[cls_id].append(score)

    # Compute running mAP proxy: mean of per-class average scores
    map_proxy = 0.0
    if len(class_scores) > 0:
        per_class_avg = [np.mean(scores_list) for scores_list in class_scores.values() if len(scores_list) > 0]
        if len(per_class_avg) > 0:
            map_proxy = float(np.mean(per_class_avg))

    # 4.4 Handle Output: draw rectangles, labels, and mAP on the frame, then write to file
    overlay = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        label = det['label']

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label text
        text = f"{label}: {score:.2f}"
        # Draw background for text for better readability
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + th + 10
        cv2.rectangle(overlay, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline),
                      (0, 255, 0), thickness=cv2.FILLED)
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)

    # Add inference time and mAP proxy on the frame
    status_text = f"Infer: {infer_ms:.1f} ms | mAP: {map_proxy:.3f}"
    cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2, cv2.LINE_AA)

    writer.write(overlay)

# Phase 5: Cleanup
cap.release()
writer.release()
cv2.destroyAllWindows()

# Print summary
if frame_count > 0:
    avg_infer_ms = total_inference_time_ms / frame_count
else:
    avg_infer_ms = 0.0

final_map = 0.0
if len(class_scores) > 0:
    per_class_avg = [np.mean(scores_list) for scores_list in class_scores.values() if len(scores_list) > 0]
    if len(per_class_avg) > 0:
        final_map = float(np.mean(per_class_avg))

print("Processing complete.")
print(f"Frames processed: {frame_count}")
print(f"Average inference time per frame: {avg_infer_ms:.2f} ms")
print(f"mAP (proxy based on average detection confidences per class): {final_map:.3f}")
print(f"Output video saved to: {output_path}")