import os
import time
import numpy as np
import cv2

# Configuration parameters
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_DELEGATE_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print(f"[WARN] Label file not found at: {path}. Class names will default to IDs.")
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Common formats:
            # 1) "0 person"
            # 2) "0: person"
            # 3) "person" (implicit index)
            if ":" in line:
                parts = line.split(":", 1)
            else:
                parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                labels[int(parts[0].strip())] = parts[1].strip()
            else:
                # Fallback: index by line number
                labels[idx] = line
    return labels


def make_interpreter(model_path, delegate_path):
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError as e:
        raise SystemExit(f"[ERROR] Failed to import tflite_runtime. Ensure it is installed on the device. Details: {e}")

    if not os.path.isfile(model_path):
        raise SystemExit(f"[ERROR] TFLite model not found: {model_path}")
    if not os.path.isfile(delegate_path):
        raise SystemExit(f"[ERROR] EdgeTPU delegate not found: {delegate_path}")

    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(delegate_path)]
        )
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to load interpreter with EdgeTPU delegate. Details: {e}")
    interpreter.allocate_tensors()
    return interpreter


def get_input_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    input_dtype = input_details['dtype']
    input_shape = input_details['shape']  # [1, height, width, 3]
    height, width = int(input_shape[1]), int(input_shape[2])
    quant_params = input_details.get('quantization', (0.0, 0))
    return input_index, (height, width), input_dtype, quant_params


def preprocess(frame_bgr, input_size, input_dtype, quant_params):
    # Resize and convert BGR to RGB
    ih, iw = input_size
    resized = cv2.resize(frame_bgr, (iw, ih))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Prepare input tensor
    if np.issubdtype(input_dtype, np.floating):
        # Normalize to [0, 1] float32 as a common default for float models
        input_tensor = rgb.astype(np.float32) / 255.0
    else:
        # Quantized input (e.g., uint8/int8) with scale/zero_point
        scale, zero_point = quant_params if quant_params is not None else (0.0, 0)
        if scale and scale > 0:
            input_tensor = rgb.astype(np.float32) / scale + zero_point
            # Round and clip to valid data type range
            if input_dtype == np.uint8:
                input_tensor = np.clip(np.rint(input_tensor), 0, 255).astype(np.uint8)
            elif input_dtype == np.int8:
                input_tensor = np.clip(np.rint(input_tensor), -128, 127).astype(np.int8)
            else:
                input_tensor = input_tensor.astype(input_dtype)
        else:
            # If quantization params are not provided, assume uint8 0-255
            input_tensor = rgb.astype(input_dtype)

    # Add batch dimension [1, h, w, 3]
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def set_input_tensor(interpreter, input_index, input_tensor):
    interpreter.set_tensor(input_index, input_tensor)


def _dequantize_if_needed(tensor, details):
    # Attempt to dequantize tensor if it has quantization parameters and is integer type
    if not np.issubdtype(tensor.dtype, np.floating):
        q = details.get('quantization_parameters', details.get('quantization', None))
        if isinstance(q, dict):
            scales = q.get('scales', None)
            zero_points = q.get('zero_points', None)
            if scales is not None and np.size(scales) > 0:
                scale = float(scales[0])
                zero_point = int(zero_points[0]) if (zero_points is not None and np.size(zero_points) > 0) else 0
                return (tensor.astype(np.float32) - zero_point) * scale
        elif isinstance(q, (tuple, list)) and len(q) == 2:
            scale, zero_point = q
            if scale and scale > 0:
                return (tensor.astype(np.float32) - zero_point) * scale
    return tensor


def extract_detections(interpreter, original_hw, conf_threshold):
    """
    Extracts detection boxes, class IDs, and scores from the interpreter outputs.
    Returns a list of detections: each is dict with keys: 'bbox' (xmin, ymin, xmax, ymax), 'score', 'class_id'
    """
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]

    # Try to identify boxes, classes, scores, count from shapes and value ranges
    boxes = None
    classes = None
    scores = None
    count = None

    # Attempt mapping by common order: boxes, classes, scores, count
    if len(outputs) >= 3:
        # Heuristic pass
        for od, out in zip(output_details, outputs):
            out_dq = _dequantize_if_needed(out, od)
            if out_dq.ndim == 3 and out_dq.shape[-1] == 4:
                boxes = out_dq
            elif out_dq.ndim == 2:
                # Could be classes or scores
                # Scores typically in [0, 1]; classes typically large ints/floats
                mx = float(np.max(out_dq)) if out_dq.size > 0 else 0.0
                mn = float(np.min(out_dq)) if out_dq.size > 0 else 0.0
                # If all values between 0 and 1, more likely scores
                if mn >= 0.0 and mx <= 1.0 and scores is None:
                    scores = out_dq
                else:
                    classes = out_dq
            elif out_dq.size == 1 and count is None:
                count = int(np.rint(out_dq.flatten()[0]))

    # Fallbacks if not determined
    if boxes is None:
        for out in outputs:
            if out.ndim == 3 and out.shape[-1] == 4:
                boxes = out
                break
    if scores is None:
        # Choose the 2D float tensor with values in [0,1]
        for out in outputs:
            if out.ndim == 2 and np.issubdtype(out.dtype, np.floating):
                if out.size > 0 and float(np.max(out)) <= 1.0 and float(np.min(out)) >= 0.0:
                    scores = out
                    break
    if classes is None:
        for out in outputs:
            if out.ndim == 2 and (np.issubdtype(out.dtype, np.integer) or (np.issubdtype(out.dtype, np.floating) and (float(np.max(out)) > 1.0 or float(np.min(out)) < 0.0))):
                classes = out
                break
        # If still None and exactly three outputs, assign whichever 2D float tensor not used as scores
        if classes is None and len(outputs) == 3:
            for out in outputs:
                if out.ndim == 2 and out is not scores:
                    classes = out
                    break
    if count is None and scores is not None:
        count = scores.shape[1]
    if count is None and boxes is not None:
        count = boxes.shape[1]

    if boxes is None or scores is None or classes is None:
        # Cannot parse outputs reliably
        return []

    # Ensure proper shapes: squeeze batch dimension [1, N, ...] -> [N, ...]
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    classes = np.squeeze(classes, axis=0)
    if classes.dtype != np.int32 and classes.dtype != np.int64:
        classes = np.rint(classes).astype(np.int32)

    h, w = original_hw
    detections = []
    N = int(count) if count is not None else boxes.shape[0]
    for i in range(N):
        score = float(scores[i])
        if score < conf_threshold:
            continue
        cls_id = int(classes[i])
        # Boxes from TFLite detection postprocess are [ymin, xmin, ymax, xmax] normalized [0..1]
        y_min, x_min, y_max, x_max = boxes[i].tolist()
        # Clip to [0,1]
        x_min = min(max(x_min, 0.0), 1.0)
        y_min = min(max(y_min, 0.0), 1.0)
        x_max = min(max(x_max, 0.0), 1.0)
        y_max = min(max(y_max, 0.0), 1.0)

        xmin_px = int(x_min * w)
        ymin_px = int(y_min * h)
        xmax_px = int(x_max * w)
        ymax_px = int(y_max * h)

        # Sanity check and clip to image bounds
        xmin_px = max(0, min(xmin_px, w - 1))
        xmax_px = max(0, min(xmax_px, w - 1))
        ymin_px = max(0, min(ymin_px, h - 1))
        ymax_px = max(0, min(ymax_px, h - 1))
        if xmax_px <= xmin_px or ymax_px <= ymin_px:
            continue

        detections.append({
            "bbox": (xmin_px, ymin_px, xmax_px, ymax_px),
            "score": score,
            "class_id": cls_id
        })
    return detections


def draw_detections(frame_bgr, detections, labels, map_text="mAP: N/A"):
    # Draw detections on the frame
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        score = det["score"]
        cls_id = det["class_id"]

        # Label lookup with 0-based or 1-based fallback
        label = labels.get(cls_id, labels.get(cls_id + 1, str(cls_id)))
        caption = f"{label}: {score:.2f}"

        # Choose color based on class id
        color = ((37 * (cls_id + 1)) % 255, (17 * (cls_id + 1)) % 255, (29 * (cls_id + 1)) % 255)

        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, 2)

        # Text background
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (xmin, ymin - th - 6), (xmin + tw + 2, ymin), color, thickness=-1)
        cv2.putText(frame_bgr, caption, (xmin + 1, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw mAP info at top-left
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)
    return frame_bgr


def main():
    print("[INFO] Starting 'TFLite object detection with TPU' on Google Coral Dev Board")
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] Labels: {LABEL_PATH}")
    print(f"[INFO] Input video: {INPUT_PATH}")
    print(f"[INFO] Output video: {OUTPUT_PATH}")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # Setup: Interpreter with EdgeTPU delegate and labels
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_DELEGATE_PATH)
    input_index, input_size, input_dtype, quant_params = get_input_details(interpreter)
    labels = load_labels(LABEL_PATH)

    # Input video
    if not os.path.isfile(INPUT_PATH):
        raise SystemExit(f"[ERROR] Input video not found: {INPUT_PATH}")
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Failed to open input video: {INPUT_PATH}")

    # Output writer setup
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps is None or in_fps <= 0 or np.isnan(in_fps):
        in_fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"[ERROR] Failed to open VideoWriter for: {OUTPUT_PATH}")

    # Since ground-truth is not provided, mAP cannot be computed; show as N/A.
    # The pipeline is ready for mAP overlay as text.
    map_text = "mAP: N/A (no ground truth)"

    frame_count = 0
    avg_inference_ms = None
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            t0 = time.time()
            input_tensor = preprocess(frame, input_size, input_dtype, quant_params)
            set_input_tensor(interpreter, input_index, input_tensor)
            # Inference
            t1 = time.time()
            interpreter.invoke()
            t2 = time.time()

            preprocess_ms = (t1 - t0) * 1000.0
            infer_ms = (t2 - t1) * 1000.0

            if avg_inference_ms is None:
                avg_inference_ms = infer_ms
            else:
                # Exponential moving average for smoother FPS
                avg_inference_ms = 0.9 * avg_inference_ms + 0.1 * infer_ms

            # Extract detections
            detections = extract_detections(interpreter, (frame_h, frame_w), CONFIDENCE_THRESHOLD)

            # Draw detections and overlays
            frame_out = frame.copy()
            frame_out = draw_detections(frame_out, detections, labels, map_text=map_text)

            # Show FPS (based on inference only)
            fps_text = f"TPU Inference: {infer_ms:.1f} ms (avg {avg_inference_ms:.1f} ms) | ~{(1000.0/avg_inference_ms):.1f} FPS" if avg_inference_ms and avg_inference_ms > 0 else f"Inference: {infer_ms:.1f} ms"
            cv2.putText(frame_out, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 190, 255), 2, cv2.LINE_AA)

            writer.write(frame_out)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t_start
    print(f"[INFO] Processed {frame_count} frames in {elapsed:.2f}s. Output saved to: {OUTPUT_PATH}")
    print("[INFO] Completed.")


if __name__ == "__main__":
    main()