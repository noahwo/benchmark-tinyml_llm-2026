import os
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports: Try tflite_runtime first, fallback to tensorflow.lite
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_source = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        tflite_source = "tensorflow.lite"
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to import TFLite runtime and TensorFlow Lite: {e}")

# 1.2 Paths/Parameters (from Configuration Parameters)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

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
        print(f"WARNING: Could not load labels from {label_file_path}: {e}")
    return labels

labels_list = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU
def make_interpreter_with_edgetpu(model_file_path):
    last_error = None
    try:
        # First attempt (default lib name)
        return Interpreter(model_path=model_file_path,
                           experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    except Exception as e1:
        last_error = e1
        # Second attempt (explicit absolute path on aarch64 platforms)
        try:
            return Interpreter(model_path=model_file_path,
                               experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')])
        except Exception as e2:
            err_msg = (
                "ERROR: Failed to load EdgeTPU delegate.\n"
                f"- First attempt error: {e1}\n"
                f"- Second attempt error: {e2}\n"
                "Please ensure the Coral EdgeTPU runtime is installed and the model is compiled for EdgeTPU.\n"
                "On Coral Dev Board, you can install/update the runtime via 'sudo apt-get install libedgetpu1-std' "
                "or 'libedgetpu1-max' depending on your needs."
            )
            raise RuntimeError(err_msg) from e2

try:
    interpreter = make_interpreter_with_edgetpu(model_path)
except RuntimeError as de:
    raise SystemExit(str(de))

try:
    interpreter.allocate_tensors()
except Exception as e:
    raise SystemExit(f"ERROR: Failed to allocate tensors for the interpreter: {e}")

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    raise SystemExit("ERROR: Model has no input tensors.")

# Assuming single input tensor for detection models
in_height, in_width = input_details[0]['shape'][1], input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Utility: parse detection model outputs robustly
def get_detection_outputs(interpreter, output_details):
    # Typical SSD detection models have 4 outputs: boxes, classes, scores, count
    # We will identify them by shapes and value ranges.
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    # Squeeze leading batch dimensions
    sq_outputs = [np.squeeze(o) for o in outputs]

    boxes = None
    classes = None
    scores = None
    count = None

    # Identify boxes by last dimension 4
    for o in sq_outputs:
        if o.ndim == 2 and o.shape[1] == 4:
            boxes = o
            break
    # Identify count as scalar or shape (1,)
    for o in sq_outputs:
        if o.ndim == 0 or (o.ndim == 1 and o.shape[0] == 1):
            count = int(np.round(float(np.squeeze(o))))
            break
    # Remaining two are classes and scores
    remaining = [o for o in sq_outputs if o is not boxes and o is not count]
    # If we already found boxes and count, remaining should be 2 arrays
    if len(remaining) == 2:
        a, b = remaining
        # scores are typically floats in [0,1]
        a_max = float(np.max(a)) if a.size else 0.0
        b_max = float(np.max(b)) if b.size else 0.0
        if a.dtype == np.float32 and a_max <= 1.0 + 1e-6:
            scores, classes = a, b
        elif b.dtype == np.float32 and b_max <= 1.0 + 1e-6:
            scores, classes = b, a
        else:
            # Fallback to assuming standard order if ambiguous
            scores, classes = a, b
    else:
        # Fallback to common TFLite detection output order if above heuristic fails
        try:
            boxes = np.squeeze(outputs[0])
            classes = np.squeeze(outputs[1])
            scores = np.squeeze(outputs[2])
            count = int(np.squeeze(outputs[3]))
        except Exception as e:
            raise RuntimeError(f"ERROR: Unable to parse detection outputs: {e}")

    # Ensure proper shapes
    if boxes is None or classes is None or scores is None or count is None:
        raise RuntimeError("ERROR: Detection outputs incomplete (boxes/classes/scores/count missing).")

    # Convert classes to int
    classes = classes.astype(np.int32, copy=False)
    # Handle potential over-count
    num_dets = min(count, boxes.shape[0], classes.shape[0], scores.shape[0])
    return boxes[:num_dets], classes[:num_dets], scores[:num_dets], num_dets

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire Input Data: open video file
if not os.path.exists(input_path):
    raise SystemExit(f"ERROR: Input video file not found: {input_path}")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise SystemExit(f"ERROR: Failed to open input video: {input_path}")

# Prepare output video writer with same frame size and fps
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0 or np.isnan(fps):
    fps = 30.0  # fallback to a reasonable default if fps is not available

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
if not writer.isOpened():
    cap.release()
    raise SystemExit(f"ERROR: Failed to open output video for writing: {output_path}")

# 2.2 Preprocess helper: prepare input tensor from BGR frame
def preprocess_frame_bgr_to_model_input(frame_bgr, target_w, target_h, floating):
    # Resize
    resized = cv2.resize(frame_bgr, (target_w, target_h))
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Expand dims to match [1, H, W, 3]
    input_tensor = np.expand_dims(rgb, axis=0)
    # 2.3 Quantization Handling
    if floating:
        input_tensor = (np.float32(input_tensor) - 127.5) / 127.5
    else:
        input_tensor = np.asarray(input_tensor, dtype=np.uint8)
    return input_tensor

# 2.4 Loop Control: process single video file until end
frame_index = 0
last_time = time.time()
fps_smooth = fps  # initialize smoothed FPS with nominal fps
# For mAP proxy (since no ground truth is provided), we accumulate per-class scores
per_class_scores = dict()

# =========================
# Processing Loop: Phases 2 -> 4
# =========================

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_index += 1

    # Phase 2: Preprocess
    input_tensor = preprocess_frame_bgr_to_model_input(frame_bgr, in_width, in_height, floating_model)

    # =========================
    # Phase 3: Inference
    # =========================
    try:
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
    except Exception as e:
        # Try to handle potential types/shapes mismatch explicitly
        raise SystemExit(f"ERROR: Failed to set input tensor: {e}")
    try:
        interpreter.invoke()
    except Exception as e:
        raise SystemExit(f"ERROR: Inference failed during interpreter.invoke(): {e}")

    # =========================
    # Phase 4: Output Interpretation & Handling Loop
    # =========================

    # 4.1 Get Output Tensor(s)
    try:
        det_boxes, det_classes, det_scores, det_count = get_detection_outputs(interpreter, output_details)
    except Exception as e:
        raise SystemExit(str(e))

    # 4.2 Interpret Results: build detection list with label mapping
    detections = []
    for i in range(det_count):
        score = float(det_scores[i])
        class_id = int(det_classes[i])
        # Map class id to label (labels may be zero-based; use safe lookup)
        if 0 <= class_id < len(labels_list):
            class_name = labels_list[class_id]
        else:
            class_name = f"class_{class_id}"
        # Normalized box coordinates (ymin, xmin, ymax, xmax)
        y_min, x_min, y_max, x_max = det_boxes[i].tolist()
        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "score": score,
            "box_norm": (y_min, x_min, y_max, x_max)
        })

    # 4.3 Post-processing: apply confidence thresholding, coordinate scaling, and clipping
    filtered_detections = []
    for det in detections:
        if det["score"] < confidence_threshold:
            continue
        y_min, x_min, y_max, x_max = det["box_norm"]

        # Clip to [0,1]
        y_min = float(np.clip(y_min, 0.0, 1.0))
        x_min = float(np.clip(x_min, 0.0, 1.0))
        y_max = float(np.clip(y_max, 0.0, 1.0))
        x_max = float(np.clip(x_max, 0.0, 1.0))

        # Scale to image size
        x1 = int(round(x_min * orig_width))
        y1 = int(round(y_min * orig_height))
        x2 = int(round(x_max * orig_width))
        y2 = int(round(y_max * orig_height))

        # Ensure coordinates are within frame bounds
        x1 = int(np.clip(x1, 0, orig_width - 1))
        y1 = int(np.clip(y1, 0, orig_height - 1))
        x2 = int(np.clip(x2, 0, orig_width - 1))
        y2 = int(np.clip(y2, 0, orig_height - 1))

        # Ensure proper ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        det_out = {
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "score": det["score"],
            "box": (x1, y1, x2, y2)
        }
        filtered_detections.append(det_out)

        # Update proxy mAP stats (using detection scores as proxy; true mAP needs ground truth)
        cid = det["class_id"]
        if cid not in per_class_scores:
            per_class_scores[cid] = []
        per_class_scores[cid].append(det["score"])

    # Compute proxy mAP as mean of per-class mean scores (note: this is NOT true mAP without GT)
    if len(per_class_scores) > 0:
        per_class_ap_proxy = [float(np.mean(scores)) for scores in per_class_scores.values() if len(scores) > 0]
        mAP_proxy = float(np.mean(per_class_ap_proxy)) if len(per_class_ap_proxy) > 0 else 0.0
    else:
        mAP_proxy = 0.0

    # 4.4 Handle Output: draw results on frame and write to output video
    display_frame = frame_bgr.copy()

    # Draw detections
    for det in filtered_detections:
        x1, y1, x2, y2 = det["box"]
        label_text = f"{det['class_name']} {det['score']:.2f}"
        color = (0, 255, 0)  # Green for detected boxes
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        # Text background
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - baseline - 2)
        tx1 = max(0, x1)
        cv2.rectangle(display_frame, (tx1, ty1), (tx1 + tw + 4, ty1 + th + baseline + 4), (0, 0, 0), -1)
        cv2.putText(display_frame, label_text, (tx1 + 2, ty1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Update and draw FPS
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0:
        fps_inst = 1.0 / dt
        # Smooth FPS: exponential moving average
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst

    hud_text_1 = f"FPS: {fps_smooth:.2f}"
    hud_text_2 = f"mAP*: {mAP_proxy:.3f}"  # Proxy since GT is not provided
    cv2.putText(display_frame, hud_text_1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
    cv2.putText(display_frame, hud_text_2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
    cv2.putText(display_frame, "*Proxy mAP (no GT)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    writer.write(display_frame)

# =========================
# Phase 5: Cleanup
# =========================
cap.release()
writer.release()

# Final summary
if len(per_class_scores) > 0:
    final_per_class_ap_proxy = [float(np.mean(scores)) for scores in per_class_scores.values() if len(scores) > 0]
    final_mAP_proxy = float(np.mean(final_per_class_ap_proxy)) if len(final_per_class_ap_proxy) > 0 else 0.0
else:
    final_mAP_proxy = 0.0

print(f"Processing complete.")
print(f"Output saved to: {output_path}")
print(f"Final mAP* (proxy, no ground truth): {final_mAP_proxy:.4f}")
print(f"TFLite source used: {tflite_source}")