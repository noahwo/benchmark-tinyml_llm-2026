import os
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_source = "tflite_runtime"
except Exception:
    # Fallback to TensorFlow Lite if tflite_runtime is unavailable
    from tensorflow.lite import Interpreter  # type: ignore
    try:
        from tensorflow.lite.experimental import load_delegate  # type: ignore
    except Exception as e:
        print("ERROR: Could not import EdgeTPU delegate loader. Ensure TensorFlow Lite is installed properly.")
        raise
    tflite_source = "tensorflow.lite"

# Import OpenCV only because we process video
import cv2


def load_labels(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                label = line.strip()
                if label != "":
                    labels.append(label)
    except Exception as e:
        print(f"WARNING: Failed to load labels from '{label_file_path}': {e}")
        labels = []
    return labels


def make_interpreter_with_edgetpu(model_file_path):
    # 1.4 Load Interpreter with EdgeTPU and handle errors
    # Try standard shared object name first
    try:
        interpreter = Interpreter(
            model_path=model_file_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
        print("INFO: EdgeTPU delegate loaded (libedgetpu.so.1.0) using", tflite_source)
        return interpreter, True
    except Exception as e_primary:
        print(f"WARNING: Could not load EdgeTPU delegate 'libedgetpu.so.1.0': {e_primary}")
        # Try explicit path for aarch64 systems commonly used on Coral Dev Board
        try:
            interpreter = Interpreter(
                model_path=model_file_path,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
            )
            print("INFO: EdgeTPU delegate loaded (/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0) using", tflite_source)
            return interpreter, True
        except Exception as e_secondary:
            print(f"ERROR: Failed to load EdgeTPU delegate from both default and explicit paths.\n"
                  f"Primary error: {e_primary}\nSecondary error: {e_secondary}\n"
                  f"Falling back to CPU-only TFLite interpreter. Performance will be significantly reduced.")
            try:
                interpreter = Interpreter(model_path=model_file_path)
                return interpreter, False
            except Exception as e_cpu:
                print(f"FATAL: Could not create TFLite interpreter for model '{model_file_path}': {e_cpu}")
                raise


def preprocess_frame_bgr_to_input(frame_bgr, input_shape, input_dtype, floating_model):
    # Convert BGR (OpenCV) to RGB and resize to model input size
    _, in_h, in_w, _ = input_shape  # [1, height, width, channels]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)

    if floating_model:
        # Normalize to [-1, 1] as per guideline
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Ensure uint8 input for quantized models
        if input_dtype == np.uint8:
            input_data = np.uint8(input_data)
        else:
            input_data = input_data.astype(input_dtype, copy=False)
    return input_data


def draw_detections_on_frame(frame_bgr, detections, labels, map_text):
    # Draw rectangles and labels on frame, plus global mAP text
    for det in detections:
        ymin, xmin, ymax, xmax = det['bbox']
        class_id = det['class_id']
        score = det['score']
        label_text = labels[class_id] if (labels and 0 <= class_id < len(labels)) else f"id:{class_id}"
        caption = f"{label_text} {score:.2f}"

        # Draw rectangle
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put text
        cv2.rectangle(frame_bgr, (xmin, max(0, ymin - 20)), (xmin + max(80, len(caption) * 8), ymin), (0, 255, 0), -1)
        cv2.putText(frame_bgr, caption, (xmin + 4, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw mAP text at the top-left of the frame
    cv2.rectangle(frame_bgr, (5, 5), (260, 35), (0, 0, 0), -1)
    cv2.putText(frame_bgr, map_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    # 1.2 Paths/Parameters
    model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
    label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
    input_path = '/home/mendel/tinyml_autopilot/data//sheeps.mp4'
    output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')

    # 1.3 Load Labels
    labels = load_labels(label_path)

    # 1.4 Load Interpreter with EdgeTPU
    try:
        interpreter, using_edgetpu = make_interpreter_with_edgetpu(model_path)
    except Exception:
        # Already printed detailed errors in make_interpreter_with_edgetpu
        raise SystemExit(1)

    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        print(f"WARNING: Expected 1 input tensor but found {len(input_details)}. Proceeding with the first one.")
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"FATAL: Cannot open input video file: {input_path}")
        raise SystemExit(1)

    # Video properties
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback to a reasonable default
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, fourcc, fps, (src_width, src_height))
    if not writer.isOpened():
        print(f"FATAL: Cannot open output video writer for: {output_path}")
        cap.release()
        raise SystemExit(1)

    # We cannot compute true mAP without ground-truth annotations. We'll annotate as N/A.
    map_text = "mAP: N/A (no ground truth)"

    frame_index = 0
    start_time = time.time()

    # 2.4 Loop Control: process frames until the video ends
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 2.2 Preprocess Data
        input_data = preprocess_frame_bgr_to_input(frame_bgr, input_shape, input_dtype, floating_model)

        # Phase 3: Inference
        # 3.1 Set Input Tensor(s)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling
        # 4.1 Get Output Tensor(s) - typical SSD outputs: boxes, classes, scores, count
        try:
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [N, 4] in [ymin, xmin, ymax, xmax], normalized
            classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)  # [N]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [N]
            count = int(interpreter.get_tensor(output_details[3]['index'])[0])
        except Exception:
            # Fallback in case of model-specific tensor ordering
            tensors = [interpreter.get_tensor(od['index']) for od in output_details]
            # Identify by shape
            boxes, classes, scores, count = None, None, None, None
            for t in tensors:
                arr = np.squeeze(t)
                if arr.ndim == 2 and arr.shape[-1] == 4:
                    boxes = arr
                elif arr.ndim == 1 and arr.size <= 300 and arr.dtype in (np.float32, np.float64, np.int32):
                    # Could be scores or classes; defer final assignment
                    pass
            # Try default order as last resort
            boxes = boxes if boxes is not None else tensors[0][0]
            classes = tensors[1][0].astype(np.int32)
            scores = tensors[2][0]
            count = int(tensors[3][0])

        # 4.2 Interpret Results and 4.3 Post-processing (thresholding, scaling, clipping)
        detections = []
        for i in range(count):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            ymin_rel, xmin_rel, ymax_rel, xmax_rel = boxes[i]
            # Scale to absolute pixel coordinates
            xmin = int(max(0, min(src_width - 1, xmin_rel * src_width)))
            xmax = int(max(0, min(src_width - 1, xmax_rel * src_width)))
            ymin = int(max(0, min(src_height - 1, ymin_rel * src_height)))
            ymax = int(max(0, min(src_height - 1, ymax_rel * src_height)))

            # Ensure proper box format
            if xmax <= xmin or ymax <= ymin:
                continue

            class_id = int(classes[i])
            detections.append({
                'bbox': (ymin, xmin, ymax, xmax),
                'class_id': class_id,
                'score': score
            })

        # 4.4 Handle Output: Draw and write to output video
        draw_detections_on_frame(frame_bgr, detections, labels, map_text)
        writer.write(frame_bgr)

        frame_index += 1

    elapsed = time.time() - start_time
    if frame_index > 0 and elapsed > 0:
        print(f"INFO: Processed {frame_index} frames in {elapsed:.2f}s ({frame_index / elapsed:.2f} FPS).")
    else:
        print("INFO: No frames processed.")

    # Phase 5: Cleanup
    cap.release()
    writer.release()
    print(f"INFO: Output video saved to: {output_path}")
    if not using_edgetpu:
        print("WARNING: Inference ran without EdgeTPU acceleration. Install and enable EdgeTPU for optimal performance.")
    print("INFO: mAP cannot be computed without ground-truth annotations. Annotated as 'N/A' in the output video.")


if __name__ == "__main__":
    main()