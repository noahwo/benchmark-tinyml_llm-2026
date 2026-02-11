import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5
edgetpu_lib = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# =========================
# Utility Functions
# =========================
def load_labels(path):
    # Supports "id label" or one-label-per-line formats
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if parts[0].isdigit():
                lab_id = int(parts[0])
                lab_name = parts[1].strip() if len(parts) > 1 else str(lab_id)
                labels[lab_id] = lab_name
            else:
                labels[idx] = line
                idx += 1
    return labels

def make_interpreter(model_path, edgetpu_lib_path):
    return Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(edgetpu_lib_path)]
    )

def resolve_output_indices(output_details, label_count_hint=None):
    boxes_idx = None
    classes_idx = None
    scores_idx = None
    count_idx = None

    # Identify by name/shape first
    for i, d in enumerate(output_details):
        name = str(d.get('name', '')).lower()
        shape = d['shape']
        if (len(shape) == 3 and shape[-1] == 4) or (len(shape) == 2 and shape[-1] == 4):
            boxes_idx = i
        elif np.prod(shape) == 1:
            count_idx = i
        else:
            if 'score' in name:
                scores_idx = i
            if 'class' in name:
                classes_idx = i

    # Fallback: choose remaining 1D arrays for classes/scores
    candidates = [i for i in range(len(output_details)) if i not in [boxes_idx, count_idx] and len(output_details[i]['shape']) == 2]
    if scores_idx is None and candidates:
        # Prefer float arrays for scores
        float_cands = [i for i in candidates if output_details[i]['dtype'] == np.float32]
        scores_idx = float_cands[0] if float_cands else candidates[0]
    if classes_idx is None:
        # Remaining after picking scores
        rem = [i for i in candidates if i != scores_idx]
        classes_idx = rem[0] if rem else classes_idx

    # Additional sanity: try to swap if classes look like > label count while scores in [0,1]
    if classes_idx is not None and scores_idx is not None:
        # We will check post-inference and swap dynamically if needed
        pass

    return boxes_idx, classes_idx, scores_idx, count_idx

def preprocess_frame(frame, input_w, input_h, dtype, quant_params=None):
    # Resize and convert color space
    resized = cv2.resize(frame, (input_w, input_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if dtype == np.uint8:
        input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
        # If quantization parameters are provided, we typically still pass uint8 0-255
        # as most TFLite quantized models expect that dynamic range.
    else:
        # float32
        input_data = (np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0)

    return input_data

def color_for_class(cid):
    # Deterministic pseudo-color for a class id
    r = (37 * cid) % 255
    g = (17 * cid + 99) % 255
    b = (29 * cid + 199) % 255
    return int(b), int(g), int(r)

def compute_running_map(class_scores):
    # Pseudo-mAP: mean of per-class average confidence over detections >= threshold
    if not class_scores:
        return 0.0
    aps = []
    for _, scores in class_scores.items():
        if scores:
            aps.append(float(np.mean(scores)))
    if not aps:
        return 0.0
    return float(np.mean(aps))

# =========================
# Main Application
# =========================
def main():
    # 1. Setup
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found at: " + model_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input video not found at: " + input_path)

    labels = load_labels(label_path)
    label_count_hint = len(labels) if labels else None

    interpreter = make_interpreter(model_path, edgetpu_lib)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input tensor details
    input_idx = input_details[0]['index']
    input_shape = input_details[0]['shape']
    # Expected [1, height, width, 3]
    in_h = int(input_shape[1])
    in_w = int(input_shape[2])
    in_dtype = input_details[0]['dtype']
    in_quant = input_details[0].get('quantization', (0.0, 0))

    # Resolve outputs
    boxes_idx, classes_idx, scores_idx, count_idx = resolve_output_indices(output_details, label_count_hint)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video: " + input_path)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    # Prepare output writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (src_w, src_h))
    if not writer.isOpened():
        # Fallback: try a different codec
        fourcc_alt = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc_alt, fps, (src_w, src_h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open output video writer at: " + output_path)

    # For mAP computation (pseudo)
    class_scores = {}
    frame_index = 0
    prev_time = time.time()

    # 2-4. Processing loop: Preprocess -> Inference -> Output handling
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Preprocess
        input_data = preprocess_frame(frame, in_w, in_h, in_dtype, in_quant)

        # Inference
        interpreter.set_tensor(input_idx, input_data)
        interpreter.invoke()

        # Postprocess - read outputs
        def get_tensor_by_idx(idx):
            return interpreter.get_tensor(output_details[idx]['index']) if idx is not None else None

        boxes = get_tensor_by_idx(boxes_idx)
        classes = get_tensor_by_idx(classes_idx)
        scores = get_tensor_by_idx(scores_idx)
        count = get_tensor_by_idx(count_idx)

        # Normalize output shapes
        if boxes is not None and boxes.ndim == 3:
            boxes = boxes[0]
        if classes is not None and classes.ndim >= 2:
            classes = classes[0]
        if scores is not None and scores.ndim >= 2:
            scores = scores[0]
        if count is not None:
            n = int(np.squeeze(count))
        else:
            # Infer N from available tensors
            n = 0
            if scores is not None:
                n = scores.shape[0]
            elif boxes is not None:
                n = boxes.shape[0]
            elif classes is not None:
                n = classes.shape[0]

        # Sometimes classes/scores indices might be swapped; fix if needed
        if classes is not None and scores is not None:
            # If classes look like probabilities and scores look like IDs, swap
            if np.max(classes) <= 1.01 and np.max(scores) > 1.01:
                classes, scores = scores, classes

        # Draw detections and collect scores
        if boxes is None or classes is None or scores is None:
            # If model outputs do not conform, just write frame
            writer.write(frame)
            continue

        for i in range(n):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            # Class ID
            cid_raw = classes[i]
            try:
                cid = int(cid_raw)
            except Exception:
                cid = int(np.round(cid_raw))

            # Box coordinates (ymin, xmin, ymax, xmax) normalized [0,1]
            y_min, x_min, y_max, x_max = boxes[i]
            left = int(max(0, min(src_w - 1, x_min * src_w)))
            right = int(max(0, min(src_w - 1, x_max * src_w)))
            top = int(max(0, min(src_h - 1, y_min * src_h)))
            bottom = int(max(0, min(src_h - 1, y_max * src_h)))

            # Label text
            label_str = labels.get(cid, f"id={cid}")
            text = f"{label_str} {score*100:.1f}%"

            # Draw rectangle and label
            color = color_for_class(cid)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # Text background
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (left, max(0, top - th - 6)), (left + tw + 2, top), color, -1)
            cv2.putText(frame, text, (left + 1, max(0, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Collect scores for pseudo-mAP
            if cid not in class_scores:
                class_scores[cid] = []
            class_scores[cid].append(score)

        # Compute running pseudo-mAP
        map_value = compute_running_map(class_scores)

        # FPS estimate
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps_est = 1.0 / dt if dt > 0 else 0.0

        # Overlay summary text
        summary = f"mAP: {map_value:.3f} | FPS: {fps_est:.1f}"
        cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Write to output
        writer.write(frame)

    # Final summary frame appended optionally (not required)
    # Cleanup
    cap.release()
    writer.release()

    # Print final pseudo-mAP
    final_map = compute_running_map(class_scores)
    print("Processing completed.")
    print(f"Output saved to: {output_path}")
    print(f"Final mAP (pseudo): {final_map:.4f}")

if __name__ == "__main__":
    main()