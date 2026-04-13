import os
import sys
import time
import numpy as np

# Phase 1.1: Imports with fallback for TFLite runtime
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_source = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
        tflite_source = "tensorflow.lite"
    except Exception as e:
        print("ERROR: Failed to import TFLite Interpreter. Ensure tflite_runtime or TensorFlow Lite is installed.")
        sys.exit(1)

# Optional import only if image/video processing is explicitly needed
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is required for video processing but is not available.")
    sys.exit(1)

# Phase 1.2: Paths/Parameters
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Utility: Load labels (Phase 1.3)
def load_labels(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"WARNING: Failed to load labels from {label_file_path}: {e}")
    return labels

# Utility: EdgeTPU delegate loading with robust error handling (Phase 1.4)
def make_interpreter_with_edgetpu(model_file_path):
    last_error = None
    # Try default shared object name
    try:
        interpreter = Interpreter(
            model_path=model_file_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
        return interpreter
    except Exception as e1:
        last_error = e1
        # Try full path used on many ARM64 systems
        try:
            interpreter = Interpreter(
                model_path=model_file_path,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
            )
            return interpreter
        except Exception as e2:
            print("ERROR: Failed to load EdgeTPU delegate for the model.")
            print(f"- First attempt error (libedgetpu.so.1.0): {e1}")
            print(f"- Second attempt error (/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0): {e2}")
            print("HINTS:")
            print("  * Ensure the Google Coral EdgeTPU runtime is installed.")
            print("  * Verify the EdgeTPU is connected and recognized by the system.")
            print("  * Confirm the correct EdgeTPU shared object path for your platform.")
            sys.exit(1)

# Utility: Map output tensors by name (handles different TFLite output ordering)
def get_detection_tensors(interpreter, output_details):
    # Try to detect by tensor names
    out_tensors = {}
    name_map = {"boxes": None, "classes": None, "scores": None, "num": None}
    for od in output_details:
        name = od.get('name', '').lower()
        if 'box' in name:
            name_map['boxes'] = od['index']
        elif 'class' in name:
            name_map['classes'] = od['index']
        elif 'score' in name:
            name_map['scores'] = od['index']
        elif 'num' in name:
            name_map['num'] = od['index']

    # Fallback: deduce by shapes if names are not informative
    if any(v is None for v in name_map.values()):
        for od in output_details:
            shape = od['shape']
            if len(shape) == 3 and shape[-1] == 4:
                name_map['boxes'] = od['index']
            elif len(shape) == 2 and shape[0] == 1:
                # scores/classes: distinguish by value range after we read them
                # We'll assign temporarily; we'll disambiguate after reading.
                if name_map['scores'] is None:
                    name_map['scores'] = od['index']
                elif name_map['classes'] is None:
                    name_map['classes'] = od['index']
            elif len(shape) == 1 and shape[0] == 1:
                name_map['num'] = od['index']

    # Read tensors
    boxes = interpreter.get_tensor(name_map['boxes']) if name_map['boxes'] is not None else None
    t2 = interpreter.get_tensor(name_map['scores']) if name_map['scores'] is not None else None
    t3 = interpreter.get_tensor(name_map['classes']) if name_map['classes'] is not None else None
    num = interpreter.get_tensor(name_map['num']) if name_map['num'] is not None else None

    # If mis-assigned, fix by checking value ranges (classes often are small integers; scores are [0,1])
    if t2 is not None and t3 is not None:
        # t2: scores candidate, t3: classes candidate
        # Determine which looks like scores by checking if values are in [0,1]
        if np.max(t2) <= 1.0 and np.min(t2) >= 0.0:
            scores = t2
            classes = t3
        elif np.max(t3) <= 1.0 and np.min(t3) >= 0.0:
            scores = t3
            classes = t2
        else:
            # Default to t2 as scores
            scores = t2
            classes = t3
    else:
        scores = t2
        classes = t3

    return boxes, classes, scores, num

# Utility: Clip and scale bounding boxes to image size (Phase 4.3)
def scale_and_clip_box(box, img_w, img_h):
    # box format: [ymin, xmin, ymax, xmax] normalized (0..1)
    ymin = max(0.0, min(1.0, float(box[0])))
    xmin = max(0.0, min(1.0, float(box[1])))
    ymax = max(0.0, min(1.0, float(box[2])))
    xmax = max(0.0, min(1.0, float(box[3])))

    x1 = int(xmin * img_w)
    y1 = int(ymin * img_h)
    x2 = int(xmax * img_w)
    y2 = int(ymax * img_h)

    # Clip to image boundaries
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2

# Utility: Compute a proxy mAP (since ground truth is unavailable here)
# We define proxy mAP as the mean of per-class average detection scores for detections above threshold.
def compute_proxy_map(class_to_scores_dict):
    ap_values = []
    for cls_id, scores in class_to_scores_dict.items():
        if len(scores) > 0:
            ap_values.append(float(np.mean(scores)))
    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))

def main():
    # Load labels
    labels = load_labels(label_path)

    # Phase 1.4: Load interpreter with EdgeTPU
    interpreter = make_interpreter_with_edgetpu(model_path)
    interpreter.allocate_tensors()

    # Phase 1.5: Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]
    input_height, input_width = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # Phase 2.1: Acquire Input Data (video file)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open input video: {input_path}")
        sys.exit(1)

    # Prepare output video writer
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps is None or input_fps <= 1e-2:
        input_fps = 30.0  # fallback
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"ERROR: Failed to open output video for writing: {output_path}")
        cap.release()
        sys.exit(1)

    # Stats for proxy mAP calculation
    class_scores_accum = {}  # class_id -> list of detection scores
    total_frames = 0
    total_detections = 0
    t0_all = time.time()

    try:
        while True:
            # Phase 2.1: Read a frame from video
            ret, frame_bgr = cap.read()
            if not ret:
                break  # Phase 4.5: loop continuation (exit on end-of-file)
            total_frames += 1
            frame_h, frame_w = frame_bgr.shape[:2]

            # Phase 2.2: Preprocess Data
            # Convert BGR to RGB and resize to model input size
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

            input_data = np.expand_dims(resized_rgb, axis=0)
            # Phase 2.3: Quantization Handling
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = np.uint8(input_data)

            # Phase 3.1: Set Input Tensor(s)
            interpreter.set_tensor(input_index, input_data)

            # Phase 3.2: Run Inference
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer_end = time.time()

            # Phase 4.1: Get Output Tensor(s)
            boxes, classes, scores, num = get_detection_tensors(interpreter, output_details)

            if boxes is None or classes is None or scores is None or num is None:
                print("ERROR: Failed to retrieve detection tensors from the model outputs.")
                break

            # Squeeze batch dimension
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)
            num_detections = int(np.squeeze(num))

            # Phase 4.2 and 4.3: Interpret Results and Post-processing
            # Apply confidence thresholding and scale/clip bounding boxes
            for i in range(num_detections):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                class_id = int(classes[i]) if classes.size > i else -1
                box = boxes[i] if boxes.ndim == 2 else boxes

                x1, y1, x2, y2 = scale_and_clip_box(box, frame_w, frame_h)

                # Draw bounding box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Map class id to label
                if 0 <= class_id < len(labels):
                    label_text = labels[class_id]
                else:
                    label_text = f"id_{class_id}"

                # Compose label string
                label_str = f"{label_text}: {score:.2f}"
                # Draw label background for readability
                (text_w, text_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_text = max(0, y1 - text_h - 4)
                cv2.rectangle(frame_bgr, (x1, y_text), (x1 + text_w + 4, y_text + text_h + baseline + 4), (0, 255, 0), -1)
                cv2.putText(frame_bgr, label_str, (x1 + 2, y_text + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Accumulate detection scores for proxy mAP
                if class_id not in class_scores_accum:
                    class_scores_accum[class_id] = []
                class_scores_accum[class_id].append(score)
                total_detections += 1

            # Compute and overlay proxy mAP on the frame
            proxy_map = compute_proxy_map(class_scores_accum)
            fps = 1.0 / max(1e-6, (t_infer_end - t_infer_start))
            info_str = f"Proxy mAP: {proxy_map:.3f} | FPS: {fps:.1f}"
            cv2.putText(frame_bgr, info_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2, cv2.LINE_AA)

            # Phase 4.4: Write annotated frame to output video
            writer.write(frame_bgr)

    finally:
        # Phase 5.1: Cleanup
        cap.release()
        writer.release()

    total_time = time.time() - t0_all
    final_proxy_map = compute_proxy_map(class_scores_accum)

    # Final summary output
    print("=== Inference Summary ===")
    print(f"Model source: {tflite_source}")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Frames processed: {total_frames}")
    print(f"Detections (>= {confidence_threshold:.2f}): {total_detections}")
    print(f"Proxy mAP (mean of per-class average detection scores): {final_proxy_map:.4f}")
    print(f"Total processing time: {total_time:.2f}s, Avg FPS (including I/O): { (total_frames / total_time) if total_time > 0 else 0.0 :.2f}")

if __name__ == "__main__":
    main()