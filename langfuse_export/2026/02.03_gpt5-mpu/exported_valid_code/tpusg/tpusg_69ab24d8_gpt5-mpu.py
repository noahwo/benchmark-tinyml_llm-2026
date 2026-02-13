import os
import sys
import time
import numpy as np

# Phase 1.1: Imports (Interpreter and EdgeTPU delegate with fallback)
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    RUNTIME_SOURCE = "tflite_runtime"
except Exception as e_rt:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        RUNTIME_SOURCE = "tensorflow.lite"
    except Exception as e_tf:
        print("ERROR: Failed to import TFLite Interpreter from both tflite_runtime and tensorflow.lite.")
        print(f"tflite_runtime import error: {e_rt}")
        print(f"tensorflow.lite import error: {e_tf}")
        sys.exit(1)

# Import cv2 only because the application explicitly requires video processing
import cv2

# Phase 1.2: Paths/Parameters (from configuration)
MODEL_PATH  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
OUTPUT_PATH  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD  = 0.5


# Phase 1.3: Load Labels
def load_labels(label_path):
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"WARNING: Failed to load labels from {label_path}: {e}")
    return labels


# Phase 1.4: Load Interpreter with EdgeTPU
def make_interpreter_with_edgetpu(model_path):
    last_error = None
    # Try standard library name first
    for lib_name in ('libedgetpu.so.1.0', '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'):
        try:
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate(lib_name)]
            )
            interpreter.allocate_tensors()
            print(f"INFO: Loaded EdgeTPU delegate '{lib_name}' using {RUNTIME_SOURCE}.")
            return interpreter
        except Exception as e:
            last_error = e
            continue
    # If both attempts failed, provide informative error and exit
    print("ERROR: Unable to load EdgeTPU delegate. Ensure the Coral EdgeTPU runtime is installed.")
    print("Tried the following delegate libraries:")
    print(" - libedgetpu.so.1.0")
    print(" - /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0")
    print(f"Last error: {last_error}")
    sys.exit(1)


# Phase 1.5: Utility functions to get model details and parse outputs
def get_model_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details


def parse_detection_outputs(interpreter, output_details):
    # Retrieve all outputs
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    boxes = None
    classes = None
    scores = None
    num = None

    # Identify outputs by shape and value ranges
    for out in outputs:
        arr = out
        # Boxes: shape [1, N, 4] or [N, 4]
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr
            continue
        if arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = arr[np.newaxis, ...]
            continue
        # num_detections: single scalar
        if arr.size == 1:
            try:
                num = int(np.round(float(arr.flatten()[0])))
            except Exception:
                num = int(arr.flatten()[0])
            continue
        # scores or classes
        # Heuristic: scores are in [0,1]; classes are indices typically >= 0 and not bounded by 1
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))
        if max_val <= 1.0 and min_val >= 0.0:
            scores = arr
        else:
            classes = arr

    # Ensure shapes are [N] for classes/scores and [N,4] for boxes
    if boxes is not None and boxes.ndim == 3:
        boxes = boxes[0]
    if classes is not None and classes.ndim == 2:
        classes = classes[0]
    if scores is not None and scores.ndim == 2:
        scores = scores[0]

    # Fallback for num if not provided
    if num is None:
        if scores is not None:
            num = scores.shape[0]
        elif classes is not None:
            num = classes.shape[0]
        elif boxes is not None:
            num = boxes.shape[0]
        else:
            num = 0

    # Slice to num detections if arrays are longer
    if boxes is not None and boxes.shape[0] >= num:
        boxes = boxes[:num]
    if classes is not None and classes.shape[0] >= num:
        classes = classes[:num]
    if scores is not None and scores.shape[0] >= num:
        scores = scores[:num]

    return boxes, classes, scores, num


def draw_detections_on_frame(frame_bgr, detections, labels, map_text):
    for det in detections:
        (ymin, xmin, ymax, xmax, score, class_id) = det
        # Draw bounding box
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Compose label text
        label_str = str(class_id)
        if 0 <= class_id < len(labels):
            label_str = labels[class_id]
        text = f"{label_str}: {score:.2f}"

        # Compute text size and place a filled rectangle for readability
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x, text_y = xmin, max(0, ymin - 5)
        cv2.rectangle(frame_bgr, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), (0, 255, 0), -1)
        cv2.putText(frame_bgr, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw mAP text on the frame (top-left corner)
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr


def main():
    # Load labels
    labels = load_labels(LABEL_PATH)

    # Create interpreter with EdgeTPU delegate and allocate tensors
    interpreter = make_interpreter_with_edgetpu(MODEL_PATH)

    # Get model I/O details
    input_details, output_details = get_model_io_details(interpreter)
    input_shape = input_details[0]['shape']
    in_height, in_width = int(input_shape[1]), int(input_shape[2])
    in_dtype = input_details[0]['dtype']
    floating_model = (in_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop (video file)
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input video not found at {INPUT_PATH}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Failed to open input video: {INPUT_PATH}")
        sys.exit(1)

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback if fps not available
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    if not out_writer.isOpened():
        print(f"ERROR: Failed to open output video writer: {OUTPUT_PATH}")
        cap.release()
        sys.exit(1)

    # For demonstration, mAP requires ground truth annotations which are not provided.
    # Therefore, we will mark it as not available and include this information in the overlay.
    map_text = "mAP: N/A (requires ground truth annotations)"

    total_frames = 0
    total_inference_time = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # End of video

            total_frames += 1

            # Phase 2.2: Preprocess Data
            # Convert BGR to RGB as most TFLite models expect RGB input
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(frame_rgb, (in_width, in_height), interpolation=cv2.INTER_LINEAR)

            input_data = np.expand_dims(resized_rgb, axis=0)

            # Phase 2.3: Quantization Handling
            if floating_model:
                # Normalize to [-1, 1]
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                # For quantized models (uint8), use raw 0-255 values
                input_data = np.uint8(input_data)

            # Phase 3: Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            total_inference_time += (t1 - t0)

            # Phase 4.1: Get Output Tensors
            boxes, classes, scores, num = parse_detection_outputs(interpreter, output_details)

            # Phase 4.2: Interpret Results and Phase 4.3: Post-processing
            detections_to_draw = []
            if boxes is not None and scores is not None and classes is not None:
                for i in range(num):
                    score = float(scores[i])
                    if score < CONFIDENCE_THRESHOLD:
                        continue
                    # Boxes are typically in normalized coordinates [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = boxes[i]
                    # Clip coordinates to [0,1]
                    ymin = max(0.0, min(1.0, float(ymin)))
                    xmin = max(0.0, min(1.0, float(xmin)))
                    ymax = max(0.0, min(1.0, float(ymax)))
                    xmax = max(0.0, min(1.0, float(xmax)))

                    # Scale to pixel coordinates
                    x_min_px = int(round(xmin * frame_width))
                    y_min_px = int(round(ymin * frame_height))
                    x_max_px = int(round(xmax * frame_width))
                    y_max_px = int(round(ymax * frame_height))

                    # Clip to frame bounds
                    x_min_px = max(0, min(frame_width - 1, x_min_px))
                    y_min_px = max(0, min(frame_height - 1, y_min_px))
                    x_max_px = max(0, min(frame_width - 1, x_max_px))
                    y_max_px = max(0, min(frame_height - 1, y_max_px))

                    class_id = int(classes[i]) if classes is not None else -1

                    detections_to_draw.append((y_min_px, x_min_px, y_max_px, x_max_px, score, class_id))

            # Phase 4.4: Handle Output (draw and write to file)
            annotated_frame = draw_detections_on_frame(frame_bgr.copy(), detections_to_draw, labels, map_text)
            out_writer.write(annotated_frame)

        # Optional logging of performance
        if total_frames > 0:
            avg_inf_ms = (total_inference_time / total_frames) * 1000.0
            print(f"Processed {total_frames} frames.")
            print(f"Average inference time per frame: {avg_inf_ms:.2f} ms")
            print(f"Output saved to: {OUTPUT_PATH}")

    finally:
        # Phase 5: Cleanup
        cap.release()
        out_writer.release()


if __name__ == "__main__":
    main()