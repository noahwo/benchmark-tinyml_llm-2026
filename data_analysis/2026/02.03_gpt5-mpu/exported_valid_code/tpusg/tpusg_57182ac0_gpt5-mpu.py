import os
import sys
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports: Interpreter and EdgeTPU delegate with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
    except Exception as e:
        print("ERROR: Unable to import TFLite Interpreter. Ensure tflite_runtime or tensorflow is installed.")
        print(f"Import error detail: {e}")
        sys.exit(1)

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5
try:
    CONF_THRESHOLD = float(confidence_threshold)
except Exception:
    CONF_THRESHOLD = 0.5

# Only import cv2 when image/video processing is required
import cv2  # OpenCV is required for video I/O and drawing

def load_labels(path):
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                label = line.strip()
                if label:
                    labels.append(label)
    except Exception as e:
        print(f"WARNING: Failed to load labels from {path}. Proceeding without labels. Detail: {e}")
    return labels

def make_interpreter_with_edgetpu(model_path_str):
    # 1.4 Load Interpreter with EdgeTPU and proper error handling
    last_error = None
    for lib in ('libedgetpu.so.1.0', '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'):
        try:
            delegate = load_delegate(lib)
            interpreter = Interpreter(model_path=model_path_str, experimental_delegates=[delegate])
            interpreter.allocate_tensors()
            print(f"INFO: EdgeTPU delegate loaded using: {lib}")
            return interpreter
        except Exception as e:
            last_error = e
            continue
    print("ERROR: Failed to load EdgeTPU delegate. Please ensure the EdgeTPU runtime is installed and the device is connected.")
    print(f"Delegate loading detail: {last_error}")
    sys.exit(1)

def get_model_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details or not output_details:
        print("ERROR: Failed to retrieve model IO details.")
        sys.exit(1)
    return input_details, output_details

def preprocess_frame(frame_bgr, input_details):
    # 2.2 Preprocess Data to match model input
    # Convert BGR to RGB and resize to model input size
    _, in_h, in_w, _ = input_details[0]['shape']
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    # Add batch dimension
    input_data = np.expand_dims(resized, axis=0)

    # 2.3 Quantization Handling
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        # Normalize to [-1, 1]
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # For quantized models (commonly uint8 on EdgeTPU), pass raw uint8
        input_data = np.asarray(input_data, dtype=input_dtype)
    return input_data

def interpret_detection_outputs(output_tensors, frame_w, frame_h, labels, threshold):
    # 4.2 Interpret Results and 4.3 Post-processing (confidence thresholding, scaling, clipping)
    # Typical TFLite detection postprocess output:
    # boxes: [1, N, 4], classes: [1, N], scores: [1, N], count: [1]
    boxes, classes, scores, count = None, None, None, None

    # Identify tensors by shape/value characteristics
    candidates = output_tensors[:]
    # boxes detection
    for t in candidates:
        if isinstance(t, np.ndarray) and t.ndim == 3 and t.shape[0] == 1 and t.shape[-1] == 4:
            boxes = t
            break
    # scores and classes detection
    remaining = [t for t in candidates if t is not boxes]
    score_candidate, class_candidate = None, None
    for t in remaining:
        if isinstance(t, np.ndarray) and t.ndim == 2 and t.shape[0] == 1:
            # Heuristic: scores are in [0,1]
            if np.all((t >= 0.0) & (t <= 1.0)):
                score_candidate = t if score_candidate is None else score_candidate
            else:
                class_candidate = t if class_candidate is None else class_candidate
    scores = score_candidate
    classes = class_candidate

    # count detection (shape [1] or [1,1])
    for t in candidates:
        if isinstance(t, np.ndarray) and t.size == 1 and (t.ndim == 1 or t.ndim == 2):
            # Likely num_detections
            count = int(np.round(float(t.reshape(-1)[0])))
            break
    if count is None and scores is not None:
        count = scores.shape[1]
    if boxes is None or scores is None or classes is None or count is None:
        raise RuntimeError("Unexpected detection model outputs. Cannot parse tensors.")

    detections = []
    n = min(count, boxes.shape[1], scores.shape[1], classes.shape[1])
    for i in range(n):
        score = float(scores[0, i])
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[0, i]
        # Clip coordinates to [0,1]
        ymin = max(0.0, min(1.0, float(ymin)))
        xmin = max(0.0, min(1.0, float(xmin)))
        ymax = max(0.0, min(1.0, float(ymax)))
        xmax = max(0.0, min(1.0, float(xmax)))
        # Scale to frame size
        x0 = int(xmin * frame_w)
        y0 = int(ymin * frame_h)
        x1 = int(xmax * frame_w)
        y1 = int(ymax * frame_h)
        # Clip to frame boundaries
        x0 = max(0, min(frame_w - 1, x0))
        y0 = max(0, min(frame_h - 1, y0))
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        class_id = int(classes[0, i])
        label = labels[class_id] if (labels and 0 <= class_id < len(labels)) else f"class_{class_id}"
        detections.append({
            'bbox': (x0, y0, x1, y1),
            'score': score,
            'class_id': class_id,
            'label': label
        })
    return detections

def draw_detections(frame, detections, fps_text=None, map_text=None):
    for det in detections:
        x0, y0, x1, y1 = det['bbox']
        label = det['label']
        score = det['score']
        color = (0, 255, 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        caption = f"{label}: {score:.2f}"
        # Text background for readability
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x0, y0 - th - baseline), (x0 + tw, y0), color, -1)
        cv2.putText(frame, caption, (x0, y0 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Overlay FPS and mAP text
    y_offset = 20
    if fps_text:
        cv2.putText(frame, fps_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)
        y_offset += 25
    if map_text:
        cv2.putText(frame, map_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def ensure_output_path(path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

def main():
    # Phase 1 continued
    labels = load_labels(label_path)

    interpreter = make_interpreter_with_edgetpu(model_path)
    input_details, output_details = get_model_io_details(interpreter)

    # Extract input tensor info
    in_shape = input_details[0]['shape']
    in_dtype = input_details[0]['dtype']
    if len(in_shape) != 4 or in_shape[0] != 1 or in_shape[3] != 3:
        print(f"ERROR: Unexpected input shape {in_shape}. Expected [1, H, W, 3].")
        sys.exit(1)

    # Phase 2: Input Acquisition & Preprocessing Loop
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Unable to open input video: {input_path}")
        sys.exit(1)

    # Prepare output writer
    ensure_output_path(output_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0  # default fallback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not out_writer.isOpened():
        print(f"ERROR: Unable to open output video for writing: {output_path}")
        cap.release()
        sys.exit(1)

    # Placeholder for mAP: No ground truth provided; will annotate as N/A
    computed_map_text = "mAP: N/A (requires ground truth)"

    print("INFO: Starting inference on video...")
    frame_index = 0
    inference_times = []

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # 2.4 Loop Control: single video; stop at end

            # 2.2 Preprocess data
            input_data = preprocess_frame(frame_bgr, input_details)

            # Phase 3: Inference
            # 3.1 Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            # 3.2 Run Inference
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000.0
            inference_times.append(infer_ms)

            # Phase 4: Output Interpretation & Handling
            # 4.1 Get Output Tensor(s)
            output_tensors = [interpreter.get_tensor(od['index']) for od in output_details]

            # 4.2 and 4.3 Interpret and Post-process detections
            try:
                detections = interpret_detection_outputs(output_tensors, frame_w, frame_h, labels, CONF_THRESHOLD)
            except Exception as e:
                print(f"ERROR: Failed to interpret detection outputs at frame {frame_index}. Detail: {e}")
                detections = []

            # Prepare FPS text
            avg_infer_ms = infer_ms
            if len(inference_times) >= 5:
                # Smoothing average over last 5 frames
                avg_infer_ms = float(np.mean(inference_times[-5:]))
            fps_text = f"Inference: {avg_infer_ms:.1f} ms ({1000.0/avg_infer_ms:.1f} FPS)" if avg_infer_ms > 0 else "Inference: N/A"

            # 4.4 Handle Output: draw and write to file
            annotated = draw_detections(frame_bgr.copy(), detections, fps_text=fps_text, map_text=computed_map_text)
            out_writer.write(annotated)

            frame_index += 1

        print("INFO: Inference completed.")
        if inference_times:
            print(f"Average inference time: {float(np.mean(inference_times)):.2f} ms per frame")
        print("NOTE: mAP was not computed because no ground truth annotations were provided.")
        print(f"Output saved to: {output_path}")

    finally:
        # Phase 5: Cleanup
        cap.release()
        out_writer.release()

if __name__ == "__main__":
    main()