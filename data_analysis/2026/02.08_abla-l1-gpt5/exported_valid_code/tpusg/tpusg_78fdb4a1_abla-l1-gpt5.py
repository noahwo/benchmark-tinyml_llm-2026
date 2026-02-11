import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# Helper functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print("Label file not found:", path)
        return labels
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Support "id label" or just "label"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[idx] = line
    return labels

def build_interpreter(model_path):
    # Load EdgeTPU delegate from the specified path
    delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(delegate_path)]
    )
    interpreter.allocate_tensors()
    return interpreter

def get_input_size_dtype(interpreter):
    input_details = interpreter.get_input_details()[0]
    _, in_h, in_w, _ = input_details["shape"]
    in_dtype = input_details["dtype"]
    quant = input_details.get("quantization", (0.0, 0))
    return (in_w, in_h), in_dtype, quant

def prepare_input(frame_bgr, input_size, in_dtype, quant):
    in_w, in_h = input_size
    # Convert BGR->RGB and resize
    resized = cv2.resize(frame_bgr, (in_w, in_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if in_dtype == np.uint8:
        input_data = rgb.astype(np.uint8)
    else:
        # float input path (normalize 0..1)
        input_data = rgb.astype(np.float32) / 255.0
        # If quantization parameters exist for float models, apply if needed (rare)
        # Typically not needed because float models use scale 0, zero_point 0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def parse_detections(interpreter, frame_size, conf_thres, labels):
    """
    Parse TFLite detection model outputs robustly:
    - boxes: [1, N, 4] (ymin, xmin, ymax, xmax) normalized
    - classes: [1, N]
    - scores: [1, N]
    - count: [1]
    """
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(od["index"]) for od in output_details]

    # Initialize
    boxes_arr = None
    classes_arr = None
    scores_arr = None
    count_val = None

    # Identify arrays by shape
    one_d_candidates = []
    one_by_n_candidates = []
    for arr in outputs:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes_arr = arr
        elif arr.size == 1:
            count_val = int(arr.reshape(-1)[0])
        elif arr.ndim == 2 and arr.shape[0] == 1:
            one_by_n_candidates.append(arr)
        else:
            one_d_candidates.append(arr)

    # Determine scores vs classes among one_by_n_candidates
    def is_integer_array(a):
        a_flat = a.reshape(-1)
        if a_flat.size == 0:
            return False
        return np.all(np.abs(a_flat - np.round(a_flat)) < 1e-3)

    if len(one_by_n_candidates) == 2:
        a, b = one_by_n_candidates
        # Prefer the one constrained to [0,1] as scores
        a_min, a_max = float(np.min(a)), float(np.max(a))
        b_min, b_max = float(np.min(b)), float(np.max(b))
        a_is_score_like = (0.0 <= a_min <= 1.0) and (0.0 <= a_max <= 1.0) and not is_integer_array(a)
        b_is_score_like = (0.0 <= b_min <= 1.0) and (0.0 <= b_max <= 1.0) and not is_integer_array(b)
        if a_is_score_like and not b_is_score_like:
            scores_arr, classes_arr = a, b
        elif b_is_score_like and not a_is_score_like:
            scores_arr, classes_arr = b, a
        else:
            # Fallback to pick scores as the one with wider continuous spread
            if (a_max - a_min) >= (b_max - b_min):
                scores_arr, classes_arr = a, b
            else:
                scores_arr, classes_arr = b, a
    elif len(one_by_n_candidates) == 1:
        # Some models omit 'count'
        # Heuristically assume it's scores
        scores_arr = one_by_n_candidates[0]
        classes_arr = np.zeros_like(scores_arr)
    else:
        # Unexpected model outputs
        return []

    if count_val is None:
        count_val = scores_arr.shape[1]

    frame_h, frame_w = frame_size
    detections = []
    num = min(count_val, scores_arr.shape[1], boxes_arr.shape[1] if boxes_arr is not None else 0)
    for i in range(num):
        score = float(scores_arr[0, i])
        if score < conf_thres:
            continue
        cls_id = int(round(float(classes_arr[0, i])))
        if boxes_arr is not None:
            ymin, xmin, ymax, xmax = boxes_arr[0, i]
            # Scale to pixel coordinates
            x1 = int(max(0, min(frame_w - 1, xmin * frame_w)))
            y1 = int(max(0, min(frame_h - 1, ymin * frame_h)))
            x2 = int(max(0, min(frame_w - 1, xmax * frame_w)))
            y2 = int(max(0, min(frame_h - 1, ymax * frame_h)))
        else:
            # If no boxes (unexpected), skip
            continue
        label = labels.get(cls_id, f"id {cls_id}")
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": cls_id,
            "label": label
        })
    return detections

def draw_detections(frame, detections, palette, thickness=2):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        score = det["score"]
        label = det["label"]
        color = palette[cls_id % len(palette)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        caption = f"{label} {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def safe_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        return 30.0
    return float(fps)

def compute_running_map_approx(scores_per_class):
    # Approximate "mAP" as mean over classes of mean detection scores (since ground truth is not provided)
    means = []
    for scores in scores_per_class.values():
        if len(scores) > 0:
            means.append(float(np.mean(scores)))
    if len(means) == 0:
        return float("nan")
    return float(np.mean(means))

def main():
    # Load labels
    labels = load_labels(label_path)

    # Build interpreter with EdgeTPU delegate
    interpreter = build_interpreter(model_path)
    input_size, in_dtype, in_quant = get_input_size_dtype(interpreter)

    # Open video input
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Failed to open input video:", input_path)
        return

    in_w, in_h = input_size
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        # Fallback: read one frame to infer size
        ret, first = cap.read()
        if not ret:
            print("Failed to read from video:", input_path)
            cap.release()
            return
        frame_h, frame_w = first.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart

    fps_in = safe_fps(cap)
    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (frame_w, frame_h))
    if not writer.isOpened():
        print("Failed to open output writer:", output_path)
        cap.release()
        return

    # Color palette for drawing (BGR)
    palette = [
        (56, 56, 255),   # red-ish
        (56, 255, 56),   # green-ish
        (255, 56, 56),   # blue-ish
        (255, 224, 32),
        (0, 200, 255),
        (255, 0, 255),
        (80, 175, 76),
        (0, 128, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    # For mAP approximation (without ground truth)
    scores_per_class = {}

    # Inference loop
    frame_idx = 0
    last_time = time.time()
    fps_smooth = None

    input_index = interpreter.get_input_details()[0]["index"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_t = time.time()
        # Prepare input
        input_data = prepare_input(frame, input_size, in_dtype, in_quant)

        # Set tensor and invoke
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # Parse detections
        detections = parse_detections(interpreter, (frame.shape[0], frame.shape[1]), confidence_threshold, labels)

        # Aggregate scores for mAP approximation
        for det in detections:
            cid = det["class_id"]
            score = det["score"]
            if cid not in scores_per_class:
                scores_per_class[cid] = []
            scores_per_class[cid].append(score)

        # Draw detections
        draw_detections(frame, detections, palette, thickness=2)

        # Compute FPS (smoothed)
        end_t = time.time()
        dt = end_t - start_t
        if dt > 0:
            curr_fps = 1.0 / dt
            if fps_smooth is None:
                fps_smooth = curr_fps
            else:
                # Exponential moving average
                fps_smooth = 0.9 * fps_smooth + 0.1 * curr_fps
        else:
            curr_fps = 0.0

        # Compute mAP approximation
        map_approx = compute_running_map_approx(scores_per_class)
        map_text = "N/A" if (map_approx != map_approx) else f"{map_approx:.3f}"  # NaN check

        # Overlay runtime info
        info_text = f"FPS: {0.0 if fps_smooth is None else fps_smooth:.2f} | mAP: {map_text}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 230), 2, cv2.LINE_AA)

        # Write frame
        writer.write(frame)

        frame_idx += 1

    # Finalize
    cap.release()
    writer.release()

    # Print final mAP approximation summary
    final_map = compute_running_map_approx(scores_per_class)
    if final_map == final_map:  # not NaN
        print(f"Finished. Approximate mAP over detections (no GT): {final_map:.4f}")
    else:
        print("Finished. mAP could not be computed (no ground truth available).")

if __name__ == "__main__":
    main()