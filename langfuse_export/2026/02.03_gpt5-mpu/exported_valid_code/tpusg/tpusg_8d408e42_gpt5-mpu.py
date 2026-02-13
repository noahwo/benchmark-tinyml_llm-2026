import os
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports: Interpreter and EdgeTPU delegate with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    runtime_source = "tflite_runtime"
except Exception:
    # Fallback to TensorFlow Lite runtime if tflite_runtime is unavailable
    from tensorflow.lite import Interpreter  # type: ignore
    from tensorflow.lite.experimental import load_delegate  # type: ignore
    runtime_source = "tensorflow.lite"

# 1.2 Paths/Parameters (as provided)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Import cv2 only because image/video processing is explicitly required
import cv2


def load_labels(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    except Exception as e:
        print(f"Warning: Failed to read labels from {label_file_path}: {e}")
    return labels


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def preprocess_frame_for_model(frame_bgr, input_shape, floating_model):
    # Convert BGR (OpenCV) to RGB (model common expectation)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    _, in_h, in_w, in_c = input_shape
    resized = cv2.resize(frame_rgb, (in_w, in_h))
    input_data = np.expand_dims(resized, axis=0)

    if floating_model:
        # Normalize to [-1, 1] as per guideline when floating model is used
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.asarray(input_data, dtype=np.uint8)

    return input_data


def clip_bbox(xmin, ymin, xmax, ymax, width, height):
    xmin = max(0, min(xmin, width - 1))
    xmax = max(0, min(xmax, width - 1))
    ymin = max(0, min(ymin, height - 1))
    ymax = max(0, min(ymax, height - 1))
    return xmin, ymin, xmax, ymax


def compute_surrogate_map(class_score_dict):
    """
    Surrogate mAP calculation without ground-truth:
    - For each class, compute AP surrogate as the mean detection score (confidence) for that class.
    - mAP surrogate is the mean of these per-class means.
    Note: This is NOT a true mAP; it's a proxy since no ground truth is provided.
    """
    if not class_score_dict:
        return 0.0
    per_class_ap = []
    for scores in class_score_dict.values():
        if len(scores) > 0:
            per_class_ap.append(float(np.mean(scores)))
    if len(per_class_ap) == 0:
        return 0.0
    return float(np.mean(per_class_ap))


def main():
    # 1.3 Load Labels
    labels = load_labels(label_path)
    if labels:
        print(f"Loaded {len(labels)} labels from {label_path}")
    else:
        print("No labels loaded; class IDs will be used in output.")

    # 1.4 Load Interpreter with EdgeTPU
    interpreter = None
    edgetpu_loaded = False
    edge_error_msg = ""

    # Try default EdgeTPU delegate name
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
        edgetpu_loaded = True
        print("EdgeTPU delegate loaded: libedgetpu.so.1.0")
    except Exception as e1:
        edge_error_msg = f"{e1}"
        # Try alternative path typical on aarch64 systems
        try:
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
            )
            edgetpu_loaded = True
            print("EdgeTPU delegate loaded: /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0")
        except Exception as e2:
            edge_error_msg += f" | {e2}"
            print("Warning: Failed to load EdgeTPU delegate. Falling back to CPU.")
            print("Details:", edge_error_msg)
            try:
                interpreter = Interpreter(model_path=model_path)
            except Exception as e3:
                print("Error: Failed to create TFLite Interpreter:", e3)
                return

    # Allocate tensors
    try:
        interpreter.allocate_tensors()
    except Exception as e:
        print("Error: Failed to allocate tensors for the interpreter:", e)
        return

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # [1, height, width, channels]
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    print(f"Interpreter backend: {runtime_source}, EdgeTPU enabled: {edgetpu_loaded}")
    print(f"Model input shape: {input_shape}, dtype: {input_dtype}")

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Failed to open input video: {input_path}")
        return

    # Get input video properties
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # default fallback

    # Prepare output writer
    ensure_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (input_width, input_height))
    if not writer.isOpened():
        print(f"Error: Failed to open output video writer: {output_path}")
        cap.release()
        return

    # Variables for computing a running surrogate mAP
    class_score_accumulator = {}  # class_id -> list of scores (detections above threshold)
    total_frames = 0
    processed_frames = 0
    start_time = time.time()

    print(f"Processing video: {input_path}")
    print(f"Writing annotated output to: {output_path}")

    # 2.4 Loop: process each frame
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video
        total_frames += 1

        # 2.2 Preprocess Data according to model input
        input_data = preprocess_frame_for_model(frame_bgr, input_shape, floating_model)

        # 2.3 Quantization handling is embedded in preprocess (based on dtype)

        # Phase 3: Inference
        # 3.1 Set Input Tensor
        interpreter.set_tensor(input_index, input_data)

        # 3.2 Run Inference
        inf_start = time.time()
        interpreter.invoke()
        inf_end = time.time()
        inference_ms = (inf_end - inf_start) * 1000.0

        # Phase 4: Output Interpretation & Handling
        # 4.1 Get Output Tensors
        # Typical SSD model outputs: boxes [1, N, 4], classes [1, N], scores [1, N], num [1]
        try:
            boxes = interpreter.get_tensor(output_details[0]['index'])
            classes = interpreter.get_tensor(output_details[1]['index'])
            scores = interpreter.get_tensor(output_details[2]['index'])
            num = interpreter.get_tensor(output_details[3]['index'])
        except Exception:
            # Fallback: identify by shapes if output order differs
            boxes = classes = scores = num = None
            for od in output_details:
                tensor = interpreter.get_tensor(od['index'])
                shp = tensor.shape
                if len(shp) == 3 and shp[-1] == 4:
                    boxes = tensor
                elif len(shp) == 2 and shp[0] == 1 and tensor.dtype in (np.float32, np.float64):
                    # Could be scores or classes (classes often float on some models)
                    if 'scores' in od.get('name', '').lower():
                        scores = tensor
                    elif 'classes' in od.get('name', '').lower():
                        classes = tensor
                    else:
                        # Heuristic: classes may be float but near-integers; scores are in [0,1]
                        if np.max(tensor) <= 1.0:
                            scores = tensor
                        else:
                            classes = tensor
                elif len(shp) == 1 and shp[0] == 1:
                    num = tensor
            if boxes is None or classes is None or scores is None or num is None:
                print("Error: Unable to parse model outputs.")
                break

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        num_detections = int(np.squeeze(num))

        # Ensure proper types
        if classes.dtype != np.int32 and classes.dtype != np.int64:
            classes = classes.astype(np.int32)

        # 4.2 Interpret Results + 4.3 Post-processing (thresholding, scaling, clipping)
        h, w = frame_bgr.shape[:2]
        detected_count = 0
        for i in range(num_detections):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            # Bounding box is in normalized ymin, xmin, ymax, xmax
            ymin, xmin, ymax, xmax = boxes[i]
            x_min = int(xmin * w)
            x_max = int(xmax * w)
            y_min = int(ymin * h)
            y_max = int(ymax * h)
            x_min, y_min, x_max, y_max = clip_bbox(x_min, y_min, x_max, y_max, w, h)

            cls_id = int(classes[i])
            label = labels[cls_id] if (0 <= cls_id < len(labels)) else f"id {cls_id}"

            # Draw rectangle and label
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_text = f"{label}: {score*100:.1f}%"
            # Draw label background for readability
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x_min, max(0, y_min - text_h - baseline - 4)),
                          (x_min + text_w + 2, y_min), (0, 255, 0), thickness=-1)
            cv2.putText(frame_bgr, label_text, (x_min + 1, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Accumulate scores for surrogate mAP
            if cls_id not in class_score_accumulator:
                class_score_accumulator[cls_id] = []
            class_score_accumulator[cls_id].append(score)
            detected_count += 1

        processed_frames += 1

        # Compute running surrogate mAP and overlay metrics
        running_map = compute_surrogate_map(class_score_accumulator)
        fps_est = 1000.0 / inference_ms if inference_ms > 0 else 0.0

        overlay_text_1 = f"Inference: {inference_ms:.1f} ms, FPS~{fps_est:.1f}"
        overlay_text_2 = f"mAP (surrogate, no GT): {running_map:.3f} | Detections: {detected_count}"

        cv2.putText(frame_bgr, overlay_text_1, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, overlay_text_2, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2, cv2.LINE_AA)

        # 4.4 Handle Output: write annotated frame to output video
        writer.write(frame_bgr)

    # Phase 5: Cleanup
    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    final_map = compute_surrogate_map(class_score_accumulator)

    print("Processing complete.")
    print(f"Frames processed: {processed_frames}/{total_frames} in {elapsed:.2f} s")
    print(f"Surrogate mAP (no ground truth available): {final_map:.4f}")
    if not edgetpu_loaded:
        print("Note: Inference ran on CPU because EdgeTPU delegate could not be loaded.")
    else:
        print("Inference utilized EdgeTPU acceleration.")


if __name__ == "__main__":
    main()