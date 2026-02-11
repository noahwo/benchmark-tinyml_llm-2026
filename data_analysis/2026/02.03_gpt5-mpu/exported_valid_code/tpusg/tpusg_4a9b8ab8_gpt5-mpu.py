import os
import sys
import time
import numpy as np

# Phase 1.1: Imports (Interpreter + EdgeTPU delegate with fallback)
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
    except Exception as e:
        print("ERROR: Unable to import tflite_runtime or tensorflow.lite Interpreter.")
        print(f"Detail: {e}")
        sys.exit(1)

# Import cv2 only because image/video processing is explicitly required
try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV (cv2) is required for video I/O but could not be imported.")
    print(f"Detail: {e}")
    sys.exit(1)

# Phase 1.2: Paths/Parameters (using provided configuration parameters)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold_str  = 0.5
try:
    confidence_threshold = float(confidence_threshold_str)
except Exception:
    confidence_threshold = 0.5

# Utility: Load labels (Phase 1.3)
def load_labels(path):
    labels = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    except Exception as e:
        print(f"WARNING: Failed to load labels from {path}. Detail: {e}")
    return labels

# Phase 1.4: Load Interpreter with EdgeTPU delegate
def make_interpreter_tpu(model_file):
    last_err = None
    # Try default shared object name used on Coral
    try:
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
        return interpreter
    except Exception as e1:
        last_err = e1
    # Try absolute path used on aarch64 Linux
    try:
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
        return interpreter
    except Exception as e2:
        print("ERROR: Failed to load EdgeTPU delegate. Inference requires the EdgeTPU on the Coral Dev Board.")
        print("Attempts:")
        print(f" - load_delegate('libedgetpu.so.1.0') -> {last_err}")
        print(f" - load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0') -> {e2}")
        print("Please ensure the EdgeTPU runtime is installed and the TPU is accessible.")
        sys.exit(1)

# Helper: Parse detection outputs robustly
def parse_tflite_detection_outputs(interpreter, output_details):
    boxes = None
    classes = None
    scores = None
    num = None

    for od in output_details:
        out = interpreter.get_tensor(od['index'])
        # num detections
        if out.size == 1:
            try:
                num = int(np.squeeze(out).astype(np.int32))
            except Exception:
                num = int(np.squeeze(out))
            continue
        # boxes: [1, N, 4]
        if out.ndim == 3 and out.shape[-1] == 4:
            boxes = out[0]
            continue
        # classes or scores: [1, N]
        if out.ndim == 2 and out.shape[0] == 1:
            candidate = out[0]
            # Heuristic: scores typically in [0,1]
            if candidate.dtype.kind in ('f',):
                maxv = float(np.max(candidate)) if candidate.size else 0.0
                minv = float(np.min(candidate)) if candidate.size else 0.0
                if 0.0 <= minv and maxv <= 1.0:
                    scores = candidate
                else:
                    classes = candidate.astype(np.int32)
            else:
                classes = candidate.astype(np.int32)

    # Fallbacks if num is None
    if num is None:
        if scores is not None:
            num = scores.shape[0]
        elif boxes is not None:
            num = boxes.shape[0]
        elif classes is not None:
            num = classes.shape[0]
        else:
            num = 0

    # Ensure not None arrays for downstream logic
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.int32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)

    return boxes, classes, scores, num

# Helper: Draw detections on frame
def draw_detections_on_frame(frame_bgr, detections, labels, map_text):
    for det in detections:
        x0, y0, x1, y1, cls_id, score = det
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2)
        label_text = labels[cls_id] if 0 <= cls_id < len(labels) else f"id:{cls_id}"
        caption = f"{label_text} {score:.2f}"
        cv2.putText(frame_bgr, caption, (x0, max(y0 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Draw mAP text on the top-left corner
    cv2.rectangle(frame_bgr, (5, 5), (240, 30), (0, 0, 0), thickness=-1)
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr

# Main execution
def main():
    # Load labels (Phase 1.3)
    labels = load_labels(label_path)

    # Load interpreter with EdgeTPU (Phase 1.4)
    interpreter = make_interpreter_tpu(model_path)
    try:
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"ERROR: Failed to allocate tensors: {e}")
        sys.exit(1)

    # Get model I/O details (Phase 1.5)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape = input_details[0]['shape']  # [1, H, W, C]
    in_height, in_width = int(in_shape[1]), int(in_shape[2])
    in_dtype = input_details[0]['dtype']
    floating_model = (in_dtype == np.float32)

    # Phase 2.1: Acquire Input Data - open video file
    if not os.path.exists(input_path):
        print(f"ERROR: Input video file not found: {input_path}")
        sys.exit(1)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video file: {input_path}")
        sys.exit(1)

    # Prepare output video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback FPS
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        print(f"ERROR: Failed to open output video writer for: {output_path}")
        cap.release()
        sys.exit(1)

    # For mAP: Without ground truth annotations, true mAP cannot be computed.
    # We will annotate as "mAP: N/A (no GT)" to be informative while keeping processing functional.
    map_text = "mAP: N/A (no ground truth)"

    # Phase 2.4: Processing loop for single input video file
    frame_index = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # End of video

            frame_index += 1
            # Phase 2.2: Preprocess Data (resize and convert color to match model expectation RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(frame_rgb, (in_width, in_height), interpolation=cv2.INTER_LINEAR)
            input_data = np.expand_dims(resized_rgb, axis=0)

            # Phase 2.3: Quantization Handling
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                # Ensure dtype matches model input
                input_data = input_data.astype(in_dtype, copy=False)

            # Phase 3.1: Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Phase 3.2: Inference
            interpreter.invoke()

            # Phase 4.1: Get outputs
            boxes, classes, scores, num = parse_tflite_detection_outputs(interpreter, output_details)

            # Phase 4.2: Interpret Results
            # Convert normalized boxes to pixel coordinates and pair with labels/scores
            detections = []
            for i in range(int(num)):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score < confidence_threshold:
                    continue
                cls_id = int(classes[i]) if i < len(classes) else -1
                if i < len(boxes):
                    ymin, xmin, ymax, xmax = boxes[i]
                else:
                    ymin, xmin, ymax, xmax = 0.0, 0.0, 0.0, 0.0

                # Phase 4.3: Post-processing (scale + clip)
                x0 = int(max(0, min(orig_w - 1, round(xmin * orig_w))))
                y0 = int(max(0, min(orig_h - 1, round(ymin * orig_h))))
                x1 = int(max(0, min(orig_w - 1, round(xmax * orig_w))))
                y1 = int(max(0, min(orig_h - 1, round(ymax * orig_h))))

                # Ensure proper box orientation
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                detections.append((x0, y0, x1, y1, cls_id, score))

            # Phase 4.4: Handle Output (draw and write to file)
            annotated = draw_detections_on_frame(frame_bgr, detections, labels, map_text)
            writer.write(annotated)

            # Phase 4.5: Loop continuation automatically by while True and break on EOF

    finally:
        # Phase 5.1: Cleanup resources
        cap.release()
        writer.release()

    print(f"Processing completed. Output video with detections saved to: {output_path}")

if __name__ == "__main__":
    main()