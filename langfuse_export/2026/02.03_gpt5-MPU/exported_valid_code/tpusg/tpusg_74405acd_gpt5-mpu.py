import os
import sys
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports: Interpreter and EdgeTPU delegate with fallback paths
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_backend = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        tflite_backend = "tensorflow.lite"
    except Exception as e:
        print("ERROR: Unable to import TFLite Interpreter from either 'tflite_runtime' or 'tensorflow'.")
        print(f"Detail: {e}")
        sys.exit(1)

# Import cv2 only because video processing is explicitly required
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is required for video processing but could not be imported.")
    print(f"Detail: {e}")
    sys.exit(1)

# 1.2 Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Sanity check for files
if not os.path.isfile(model_path):
    print(f"ERROR: Model file not found at: {model_path}")
    sys.exit(1)

if not os.path.isfile(input_path):
    print(f"ERROR: Input video file not found at: {input_path}")
    sys.exit(1)

if not os.path.isfile(label_path):
    print(f"WARNING: Label file not found at: {label_path}. Proceeding without labels.")
    labels = []
else:
    # 1.3 Load Labels
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"WARNING: Failed to read labels from {label_path}. Detail: {e}")
        labels = []

# 1.4 Load Interpreter with EdgeTPU delegate
interpreter = None
delegate_error_messages = []
try:
    # First try default shared object name
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e1:
    delegate_error_messages.append(str(e1))
    try:
        # Fallback to explicit system path often used on Coral Dev Board
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e2:
        delegate_error_messages.append(str(e2))
        print("ERROR: Failed to load the EdgeTPU delegate. Ensure the EdgeTPU runtime is installed.")
        print("Tried delegates: 'libedgetpu.so.1.0' and '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'")
        print("Details:")
        for idx, msg in enumerate(delegate_error_messages, 1):
            print(f"  Attempt {idx}: {msg}")
        sys.exit(1)

# Allocate tensors
try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"ERROR: Failed to allocate tensors for the interpreter. Detail: {e}")
    sys.exit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    print("ERROR: The model does not have any input tensors.")
    sys.exit(1)

# Determine input tensor properties
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Parse confidence threshold
try:
    conf_thresh = float(confidence_threshold)
except Exception:
    conf_thresh = 0.5

def preprocess_frame(frame_bgr, target_h, target_w, floating):
    """
    Resize and convert the frame to match model input requirements.
    Returns input tensor of shape [1, H, W, 3] with appropriate dtype.
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if floating:
        # Normalize to [-1, 1]
        input_data = (resized.astype(np.float32) - 127.5) / 127.5
    else:
        # Handle quantized types
        if input_dtype == np.uint8:
            input_data = resized.astype(np.uint8)
        elif input_dtype == np.int8:
            # Quantize with scale/zero_point if available
            scale, zero_point = input_details[0].get('quantization', (0.0, 0))
            if scale is None or scale == 0.0:
                # Fallback heuristic: assume input in [0,255] -> [-128,127]
                input_data = np.clip(np.round(resized.astype(np.float32) - 128.0), -128, 127).astype(np.int8)
            else:
                # Convert from [0,255] to [0,1], then quantize
                norm = resized.astype(np.float32) / 255.0
                quantized = np.round(norm / scale + zero_point)
                input_data = np.clip(quantized, -128, 127).astype(np.int8)
        else:
            # Unknown type; attempt safe cast
            input_data = resized.astype(input_dtype)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def decode_detection_outputs(interpreter, output_details):
    """
    Robustly extract boxes, classes, scores, and num_detections from detector outputs.
    Returns:
        boxes: np.ndarray [N,4] (ymin, xmin, ymax, xmax) in normalized coords
        classes: np.ndarray [N] int
        scores: np.ndarray [N] float
        num_dets: int
    """
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    boxes = None
    classes = None
    scores = None
    num = None

    # Try by name hints first
    name_map = {od.get('name', f'out_{i}'): (i, od) for i, od in enumerate(output_details)}
    for key in name_map:
        lower = key.lower()
        idx = name_map[key][0]
        val = outputs[idx]
        if 'postprocess' in lower and 'box' in lower:
            boxes = val
        elif 'postprocess' in lower and 'class' in lower:
            classes = val
        elif 'postprocess' in lower and ('score' in lower or 'prob' in lower):
            scores = val
        elif 'num' in lower and 'detection' in lower:
            num = val

    # Fallback by shape pattern
    if boxes is None or classes is None or scores is None or num is None:
        for val in outputs:
            if val.ndim == 3 and val.shape[-1] == 4 and boxes is None:
                boxes = val
        for val in outputs:
            if val.ndim == 2 and scores is None and val.dtype == np.float32:
                # Heuristic: the larger variance array is likely the scores
                if scores is None:
                    scores = val
                else:
                    # pick the one with more unique values as scores
                    if np.unique(val).size > np.unique(scores).size:
                        scores = val
        for val in outputs:
            if val.ndim == 2 and classes is None and val.dtype in (np.float32, np.int32, np.int64):
                classes = val
        for val in outputs:
            if val.ndim == 1 and val.size == 1 and num is None:
                num = val

    # Safety checks and reshaping
    if boxes is None or classes is None or scores is None:
        raise RuntimeError("Failed to parse detection outputs: boxes/classes/scores not all found.")
    if boxes.ndim == 3:
        boxes = boxes[0]
    if classes.ndim == 2:
        classes = classes[0]
    if scores.ndim == 2:
        scores = scores[0]
    if num is None:
        num_dets = min(len(scores), len(classes), len(boxes))
    else:
        num_dets = int(num.reshape([-1])[0])

    # Clip num_dets to valid range
    num_dets = max(0, min(num_dets, min(len(scores), len(classes), len(boxes))))

    # Cast classes to int
    classes = classes.astype(np.int32)

    return boxes[:num_dets], classes[:num_dets], scores[:num_dets], num_dets

def get_label_name(class_id):
    if labels and 0 <= class_id < len(labels):
        return labels[class_id]
    return f'class_{class_id}'

def draw_detections(frame, detections, map_text):
    """
    Draw bounding boxes and labels on the frame.
    detections: list of dicts with keys: 'box'(xmin,ymin,xmax,ymax), 'score', 'class_id', 'label'
    map_text: string to overlay indicating mAP or proxy metric
    """
    for det in detections:
        (xmin, ymin, xmax, ymax) = det['box']
        label = det['label']
        score = det['score']

        # Draw rectangle
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Prepare label text
        caption = f"{label}: {score:.2f}"
        # Text background for readability
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, max(0, ymin - th - baseline - 2)), (xmin + tw + 2, ymin), (0, 255, 0), thickness=-1)
        cv2.putText(frame, caption, (xmin + 1, max(0, ymin - baseline - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw mAP/proxy text at top-left
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2, cv2.LINE_AA)

def compute_proxy_map(per_class_conf_hist):
    """
    IMPORTANT: True mAP requires ground-truth annotations.
    Since none are provided, we compute a proxy metric:
    - For each class, average its detection confidences across frames (AP_proxy = mean of confidences).
    - mAP_proxy = mean of AP_proxy over classes that had detections.
    Returns (mAP_proxy_value, per_class_AP_proxy_dict)
    """
    ap_list = []
    ap_per_class = {}
    for cid, confs in per_class_conf_hist.items():
        if len(confs) > 0:
            ap = float(np.mean(confs))
            ap_per_class[cid] = ap
            ap_list.append(ap)
    if len(ap_list) == 0:
        return 0.0, ap_per_class
    return float(np.mean(ap_list)), ap_per_class

def main():
    # Determine input tensor spatial size
    if len(input_shape) != 4 or input_shape[-1] != 3:
        print(f"ERROR: Unexpected input tensor shape: {input_shape}. Expected [1, H, W, 3].")
        return
    model_in_h = int(input_shape[1])
    model_in_w = int(input_shape[2])

    # Phase 2: Input Acquisition & Preprocessing Loop
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Unable to open input video: {input_path}")
        return

    # Determine video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        # Read one frame to determine size
        ret_probe, frame_probe = cap.read()
        if not ret_probe:
            print("ERROR: Could not read frame to determine video size.")
            cap.release()
            return
        height, width = frame_probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start

    # Prepare VideoWriter
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out_writer.isOpened():
        print(f"ERROR: Unable to open VideoWriter for: {output_path}")
        cap.release()
        return

    # Accumulate confidences per class for proxy mAP
    per_class_conf_hist = {}
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        # 2.2 Preprocess Data
        input_data = preprocess_frame(frame_bgr, model_in_h, model_in_w, floating_model).astype(input_dtype)

        # 2.3 Quantization Handling handled within preprocess_frame via dtype check

        # Phase 3: Inference
        try:
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
        except Exception as e:
            print(f"ERROR during inference on frame {frame_count}: {e}")
            break

        # Phase 4: Output Interpretation & Handling Loop
        # 4.1 Get Output Tensors
        try:
            boxes_norm, classes, scores, num_dets = decode_detection_outputs(interpreter, output_details)
        except Exception as e:
            print(f"ERROR: Failed to decode detection outputs on frame {frame_count}. Detail: {e}")
            break

        # 4.2 Interpret Results
        detections = []
        for i in range(num_dets):
            score = float(scores[i])
            class_id = int(classes[i])
            label_name = get_label_name(class_id)
            ymin, xmin, ymax, xmax = boxes_norm[i].tolist()

            # 4.3 Post-processing: threshold, scale, clip
            if score < conf_thresh:
                continue

            # Clip normalized coordinates
            ymin = min(max(ymin, 0.0), 1.0)
            xmin = min(max(xmin, 0.0), 1.0)
            ymax = min(max(ymax, 0.0), 1.0)
            xmax = min(max(xmax, 0.0), 1.0)

            # Scale to pixel coordinates
            x_min_px = int(round(xmin * width))
            y_min_px = int(round(ymin * height))
            x_max_px = int(round(xmax * width))
            y_max_px = int(round(ymax * height))

            # Ensure proper box ordering
            x_min_px, x_max_px = max(0, min(x_min_px, width - 1)), max(0, min(x_max_px, width - 1))
            y_min_px, y_max_px = max(0, min(y_min_px, height - 1)), max(0, min(y_max_px, height - 1))
            if x_max_px <= x_min_px or y_max_px <= y_min_px:
                continue  # invalid box after clipping/scaling

            det = {
                'box': (x_min_px, y_min_px, x_max_px, y_max_px),
                'score': score,
                'class_id': class_id,
                'label': label_name
            }
            detections.append(det)

            # Accumulate confidences for proxy mAP
            if class_id not in per_class_conf_hist:
                per_class_conf_hist[class_id] = []
            per_class_conf_hist[class_id].append(score)

        # Compute proxy mAP (since ground-truth is not available)
        map_proxy, _ = compute_proxy_map(per_class_conf_hist)
        map_text = f"mAP (proxy): {map_proxy:.3f}"

        # 4.4 Handle Output: draw and write frame
        draw_detections(frame_bgr, detections, map_text)
        out_writer.write(frame_bgr)

    elapsed = time.time() - start_time

    # Final proxy mAP computation
    final_map_proxy, per_class_ap_proxy = compute_proxy_map(per_class_conf_hist)
    print(f"Processing complete. Frames: {frame_count}, Time: {elapsed:.2f}s, FPS: {frame_count / max(elapsed, 1e-6):.2f}")
    print(f"Computed proxy mAP (mean of average detection confidences across classes): {final_map_proxy:.4f}")
    if per_class_ap_proxy:
        # Print per-class proxy AP summary (limited to top 10 classes by AP)
        sorted_items = sorted(per_class_ap_proxy.items(), key=lambda kv: kv[1], reverse=True)
        print("Top classes by proxy AP:")
        for idx, (cid, ap) in enumerate(sorted_items[:10], 1):
            print(f"  {idx}. {get_label_name(cid)} (id {cid}): {ap:.4f}")
    else:
        print("No detections above threshold; proxy mAP is 0.0")

    # Phase 5: Cleanup
    cap.release()
    out_writer.release()
    # No UI windows to destroy

if __name__ == "__main__":
    main()