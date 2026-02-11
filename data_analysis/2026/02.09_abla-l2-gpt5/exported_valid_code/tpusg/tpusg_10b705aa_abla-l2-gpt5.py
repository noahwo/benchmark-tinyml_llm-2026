import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# CONFIGURATION PARAMETERS
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# EdgeTPU shared library path on Google Coral Dev Board
edgetpu_lib_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


# =========================
# UTILITY FUNCTIONS
# =========================
def load_labels(path):
    """
    Loads labels from a text file.
    Supports either:
      - one label per line (index inferred by line number), or
      - lines in the format: "<id> <label...>"
    Returns a dict: {id: label}
    """
    labels = {}
    if not os.path.exists(path):
        raise FileNotFoundError("Label file not found: {}".format(path))
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Try to parse "id label..." format; fallback to enumerated lines
    parsed_any_id = False
    for i, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0].isdigit():
            idx = int(parts[0])
            label = " ".join(parts[1:]) if len(parts) > 1 else str(idx)
            labels[idx] = label
            parsed_any_id = True
        else:
            labels[i] = line  # temporary; if any "id label" parsed, this will be overridden later
    if parsed_any_id:
        # Ensure all indices present; do nothing extra
        return labels
    else:
        # Overwrite with enumerated if no explicit ids were found
        return {i: l for i, l in enumerate(lines)}


def make_interpreter(model_path, edgetpu_lib_path):
    """
    Creates and returns a TFLite Interpreter with EdgeTPU delegate if available.
    """
    try:
        delegate = load_delegate(edgetpu_lib_path)
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
    except Exception as e:
        print("Warning: Failed to load EdgeTPU delegate ({}). Falling back to CPU.".format(e))
        interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_input_details(interpreter):
    """
    Returns input details: (index, dtype, height, width, quant_scale, quant_zero_point)
    """
    input_details = interpreter.get_input_details()
    if not input_details:
        raise RuntimeError("No input details found in the interpreter.")
    d = input_details[0]
    index = d['index']
    dtype = d['dtype']
    shape = d['shape']
    # Expected shape [1, height, width, channels]
    if len(shape) != 4:
        raise RuntimeError("Unexpected input tensor shape: {}".format(shape))
    height, width = int(shape[1]), int(shape[2])
    scale, zero_point = d.get('quantization', (0.0, 0))
    return index, dtype, height, width, float(scale), int(zero_point)


def preprocess_frame_bgr(frame_bgr, input_h, input_w, input_dtype, quant_scale, quant_zero):
    """
    Preprocess OpenCV BGR frame to model input tensor:
      - convert BGR to RGB
      - resize to (input_w, input_h)
      - apply quantization or normalization based on model input type
      - add batch dimension
    Returns np.ndarray of shape [1, input_h, input_w, 3] with appropriate dtype.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        # Quantized path (uint8), apply scale and zero-point if provided
        if quant_scale and quant_scale > 0:
            # De-quantization expects: real_value = scale * (quantized_value - zero_point)
            # For quantization we invert: quantized = real_value / scale + zero_point
            # real_value currently in [0..255]
            input_data = input_data.astype(np.float32) / quant_scale + quant_zero
            input_data = np.clip(np.rint(input_data), 0, 255).astype(np.uint8)
        else:
            input_data = input_data.astype(np.uint8)
    return input_data


def extract_detections(interpreter, score_threshold, img_w, img_h, labels):
    """
    Extracts detection results (boxes, class_ids, scores) from the interpreter outputs.
    Returns a list of dicts:
      [{'bbox': (x_min, y_min, x_max, y_max), 'class_id': int, 'score': float, 'label': str}, ...]
    """
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    boxes, class_ids, scores, num = None, None, None, None

    # Try to resolve by tensor names first
    for i, od in enumerate(output_details):
        name = od.get('name', '').lower()
        arr = outputs[i]
        if 'box' in name or 'location' in name:
            boxes = arr
        elif 'class' in name:
            class_ids = arr
        elif 'score' in name:
            scores = arr
        elif 'num' in name:
            num = arr

    # Fallback: resolve by shape and value ranges if any are still None
    if boxes is None or class_ids is None or scores is None:
        for arr in outputs:
            shape = arr.shape
            if len(shape) == 3 and shape[-1] == 4 and boxes is None:
                boxes = arr
            elif len(shape) == 2 and shape[0] == 1 and scores is None:
                # Could be scores or class_ids; try value range to distinguish
                vmin, vmax = np.min(arr), np.max(arr)
                if 0.0 <= vmin and vmax <= 1.0:
                    scores = arr
                else:
                    class_ids = arr
            elif len(shape) == 1 and shape[0] == 1 and num is None:
                num = arr

    if boxes is None or class_ids is None or scores is None:
        raise RuntimeError("Unable to parse detection outputs from the model.")

    # Squeeze batch dim if present
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if class_ids.ndim == 2 and class_ids.shape[0] == 1:
        class_ids = class_ids[0]
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if num is not None:
        try:
            n = int(np.squeeze(num))
        except Exception:
            n = min(len(scores), len(boxes))
    else:
        n = min(len(scores), len(boxes))

    detections = []
    n = min(n, len(scores), len(boxes))
    for i in range(n):
        score = float(scores[i])
        if score < score_threshold:
            continue
        cls = int(class_ids[i]) if isinstance(class_ids[i], (np.integer, int, float)) else int(class_ids[i])
        # boxes are [ymin, xmin, ymax, xmax] normalized [0,1]
        y_min, x_min, y_max, x_max = boxes[i]
        x1 = max(0, min(int(x_min * img_w), img_w - 1))
        y1 = max(0, min(int(y_min * img_h), img_h - 1))
        x2 = max(0, min(int(x_max * img_w), img_w - 1))
        y2 = max(0, min(int(y_max * img_h), img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        label = labels.get(cls, str(cls))
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'class_id': cls,
            'score': score,
            'label': label
        })
    return detections


def draw_detections_on_frame(frame_bgr, detections, map_value=None):
    """
    Draws detection rectangles and labels on the frame.
    Optionally overlays mAP value (map_value in [0..1]) at top-left.
    """
    # Drawing settings
    box_color = (0, 255, 0)  # green
    text_color = (0, 0, 0)   # black text
    text_bg_color = (255, 255, 255)  # white background
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame_bgr.shape[:2]
    # scale font roughly by image size
    font_scale = max(0.4, min(1.2, w / 1280.0))
    thickness = max(1, int(round(w / 640.0)))

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        score = det['score']
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, thickness)
        display_text = "{}: {:.1f}%".format(label, score * 100.0)
        (tw, th), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        tx1, ty1 = x1, max(0, y1 - th - baseline - 4)
        tx2, ty2 = x1 + tw + 6, y1
        cv2.rectangle(frame_bgr, (tx1, ty1), (tx2, ty2), text_bg_color, -1)
        cv2.putText(frame_bgr, display_text, (x1 + 3, y1 - baseline - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

    if map_value is not None:
        map_text = "mAP: {:.2f}%".format(map_value * 100.0)
        (tw, th), baseline = cv2.getTextSize(map_text, font, font_scale * 1.1, thickness + 1)
        margin = 8
        bx1, by1 = margin, margin
        bx2, by2 = margin + tw + 10, margin + th + baseline + 10
        cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.putText(frame_bgr, map_text, (bx1 + 5, by2 - baseline - 4), font, font_scale * 1.1, (255, 255, 255), thickness + 1, cv2.LINE_AA)

    return frame_bgr


def compute_proxy_map(scores_by_class):
    """
    Computes a proxy mAP from detection confidences across classes (no ground-truth available).
    For each class: AP_class ≈ mean(confidences for that class above threshold),
    mAP ≈ mean(AP_class over classes with detections).
    Returns value in [0..1].
    """
    ap_values = []
    for cls_id, scores in scores_by_class.items():
        if not scores:
            continue
        ap_values.append(float(np.mean(scores)))
    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))


# =========================
# MAIN PIPELINE
# =========================
def main():
    # 1) Setup: interpreter, labels, input video
    interpreter = make_interpreter(model_path, edgetpu_lib_path)
    input_index, input_dtype, in_h, in_w, in_scale, in_zero = get_input_details(interpreter)
    labels = load_labels(label_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video: {}".format(input_path))

    # Read source properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    print("Input video: {} ({}x{}, {:.2f} FPS, {} frames)".format(input_path, src_w, src_h, src_fps, total_frames if total_frames > 0 else "unknown"))
    print("Model input size: {}x{}, dtype: {}, quant_scale: {}, zero_point: {}".format(in_w, in_h, input_dtype, in_scale, in_zero))

    # 2) Preprocessing + 3) Inference (First pass to collect detections and compute proxy mAP)
    detections_per_frame = []
    scores_by_class = {}  # {class_id: [scores]}
    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame_bgr(frame, in_h, in_w, input_dtype, in_scale, in_zero)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()

        detections = extract_detections(interpreter, confidence_threshold, frame.shape[1], frame.shape[0], labels)
        detections_per_frame.append(detections)
        # accumulate scores by class
        for det in detections:
            cls = det['class_id']
            score = det['score']
            scores_by_class.setdefault(cls, []).append(score)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print("Processed {} frames...".format(frame_idx))

    cap.release()
    elapsed = time.time() - t0
    fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
    print("First pass completed: {} frames in {:.2f}s ({:.2f} FPS)".format(frame_idx, elapsed, fps_proc))

    # 4) Compute mAP (proxy due to lack of ground-truth)
    map_value = compute_proxy_map(scores_by_class)
    print("Computed mAP (proxy): {:.2f}%".format(map_value * 100.0))

    # 4) Output handling: draw boxes, labels, and mAP; save to output video (Second pass)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        src_fps,
        (src_w, src_h)
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open output video writer at: {}".format(output_path))

    cap2 = cv2.VideoCapture(input_path)
    if not cap2.isOpened():
        writer.release()
        raise RuntimeError("Failed to reopen input video for rendering: {}".format(input_path))

    write_idx = 0
    t1 = time.time()
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        dets = detections_per_frame[write_idx] if write_idx < len(detections_per_frame) else []
        annotated = draw_detections_on_frame(frame, dets, map_value=map_value)
        writer.write(annotated)
        write_idx += 1
        if write_idx % 50 == 0:
            print("Rendered {} frames...".format(write_idx))

    cap2.release()
    writer.release()
    elapsed2 = time.time() - t1
    fps_render = write_idx / elapsed2 if elapsed2 > 0 else 0.0
    print("Rendering completed: {} frames in {:.2f}s ({:.2f} FPS)".format(write_idx, elapsed2, fps_render))
    print("Output saved to: {}".format(output_path))


if __name__ == "__main__":
    main()