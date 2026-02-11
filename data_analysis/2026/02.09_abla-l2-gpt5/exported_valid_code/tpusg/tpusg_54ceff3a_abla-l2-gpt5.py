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

# =========================
# Utilities
# =========================
def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        print(f"Label file not found at: {path}")
        return labels
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try formats:
            # 1) "0 label", "1 sheep"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
                continue
            # 2) "0: label"
            if ':' in line:
                left, right = line.split(':', 1)
                if left.strip().isdigit():
                    labels[int(left.strip())] = right.strip()
                    continue
            # 3) Just label per line
            labels[idx] = line
    return labels

def make_interpreter_with_edgetpu(model_path, edgetpu_lib="/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"):
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(edgetpu_lib)]
        )
    except (ValueError, OSError) as e:
        raise RuntimeError(f"Failed to load EdgeTPU delegate from {edgetpu_lib}: {e}")
    interpreter.allocate_tensors()
    return interpreter

def get_output_tensors_dict(interpreter):
    # Maps typical detection output tensors by name when available.
    outputs = {"boxes": None, "classes": None, "scores": None, "count": None}
    details = interpreter.get_output_details()
    for d in details:
        name = d.get("name", "")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        lname = name.lower()
        if "box" in lname:
            outputs["boxes"] = d
        elif "class" in lname:
            outputs["classes"] = d
        elif "score" in lname:
            outputs["scores"] = d
        elif "count" in lname or "num" in lname:
            outputs["count"] = d
    # Fallback heuristics if names are not informative
    if outputs["boxes"] is None or outputs["scores"] is None or outputs["classes"] is None:
        for d in details:
            shape = d.get("shape", [])
            if len(shape) == 3 and shape[-1] == 4:
                outputs["boxes"] = outputs["boxes"] or d
        # Collect 2D tensors [1, N]
        two_d = [d for d in details if len(d.get("shape", [])) == 2 and d["shape"][0] == 1]
        # Among 2D tensors, classes likely int/float with smaller dtype or named accordingly.
        # Try to assign based on dtype and uniqueness
        for d in two_d:
            name = d.get("name", "")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            lname = name.lower()
            if "class" in lname and outputs["classes"] is None:
                outputs["classes"] = d
            elif "score" in lname and outputs["scores"] is None:
                outputs["scores"] = d
        # Last resort: assign remaining by dtype guesses
        remaining = [d for d in two_d if d not in outputs.values()]
        for d in remaining:
            if outputs["scores"] is None and d["dtype"] == np.float32:
                outputs["scores"] = d
            elif outputs["classes"] is None:
                outputs["classes"] = d
        # Count tensor often [1]
        for d in details:
            shape = d.get("shape", [])
            if len(shape) == 1 and shape[0] == 1:
                outputs["count"] = outputs["count"] or d
    return outputs

def draw_detections(frame, detections, labels, map_value=None):
    # detections: list of dicts with keys: 'bbox' (xmin, ymin, xmax, ymax), 'score', 'class_id'
    for det in detections:
        (xmin, ymin, xmax, ymax) = det["bbox"]
        score = det["score"]
        class_id = det["class_id"]
        label = labels.get(class_id, str(class_id))
        color = (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, ymin - th - baseline), (xmin + tw, ymin), color, -1)
        cv2.putText(frame, text, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay mAP
    map_text = "mAP: N/A (no ground truth)"
    if isinstance(map_value, (float, int)):
        map_text = f"mAP: {map_value:.3f}"
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2, cv2.LINE_AA)

def preprocess_frame(frame, input_shape, input_dtype):
    # input_shape: [1, height, width, channels]
    _, in_h, in_w, in_c = input_shape
    # Resize and convert color
    resized = cv2.resize(frame, (in_w, in_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        input_data = rgb.astype(np.float32) / 255.0
    else:
        input_data = rgb.astype(input_dtype)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def postprocess_detections(interpreter, outputs, frame_size, threshold=0.5):
    # frame_size: (width, height) of original frame
    fw, fh = frame_size
    # Retrieve tensors
    def get_tensor(d):
        return interpreter.get_tensor(d["index"]) if d is not None else None

    boxes = get_tensor(outputs["boxes"])
    classes = get_tensor(outputs["classes"])
    scores = get_tensor(outputs["scores"])
    count = get_tensor(outputs["count"])

    # Normalize expected shapes
    if boxes is None or classes is None or scores is None:
        return []

    # Typical shapes: boxes [1, N, 4], classes [1, N], scores [1, N], count [1]
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    if count is not None:
        count = int(np.squeeze(count))
    else:
        # Infer count from scores/boxes length
        count = min(len(scores), len(boxes))

    detections = []
    for i in range(count):
        score = float(scores[i])
        if score < threshold:
            continue
        # TFLite boxes are [ymin, xmin, ymax, xmax] in normalized coordinates (0..1)
        y_min, x_min, y_max, x_max = boxes[i]
        xmin = max(0, min(int(x_min * fw), fw - 1))
        ymin = max(0, min(int(y_min * fh), fh - 1))
        xmax = max(0, min(int(x_max * fw), fw - 1))
        ymax = max(0, min(int(y_max * fh), fh - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        class_id = int(classes[i]) if i < len(classes) else -1
        detections.append({
            "bbox": (xmin, ymin, xmax, ymax),
            "score": score,
            "class_id": class_id
        })
    return detections

# =========================
# Main Application
# =========================
def main():
    # Setup: Load labels
    labels = load_labels(label_path)
    if not labels:
        print("Warning: Labels could not be loaded or file is empty. Class IDs will be shown instead.")

    # Setup: Load TFLite interpreter with EdgeTPU delegate
    print("Initializing TFLite Interpreter with EdgeTPU delegate...")
    interpreter = make_interpreter_with_edgetpu(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        print(f"Warning: Model has {len(input_details)} input tensors; using the first one.")

    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Map output tensors
    outputs = get_output_tensors_dict(interpreter)

    # Video IO
    print(f"Opening input video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video at {input_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 30.0  # default fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(f"Saving output to: {output_path} ({in_w}x{in_h} @ {fps:.2f} FPS)")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for output at {output_path}")

    # Inference loop
    frame_idx = 0
    t0 = time.time()
    # Placeholder for mAP (cannot be computed without ground truth)
    computed_map = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Preprocessing
            input_data = preprocess_frame(frame, input_shape, input_dtype)

            # Set input tensor
            interpreter.set_tensor(input_index, input_data)

            # Inference
            inf_start = time.time()
            interpreter.invoke()
            inf_end = time.time()

            # Postprocess
            detections = postprocess_detections(
                interpreter, outputs, frame_size=(in_w, in_h), threshold=confidence_threshold
            )

            # Draw and annotate
            annotated = frame.copy()
            draw_detections(annotated, detections, labels, map_value=computed_map)

            # Optionally overlay FPS and frame index
            inf_ms = (inf_end - inf_start) * 1000.0
            info_text = f"Frame: {frame_idx}/{total_frames if total_frames>0 else '?'}  Inference: {inf_ms:.1f} ms"
            cv2.putText(annotated, info_text, (10, in_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2, cv2.LINE_AA)

            # Write to output
            writer.write(annotated)

            # Optional: print progress every 50 frames
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames...")

    finally:
        cap.release()
        writer.release()
        elapsed = time.time() - t0
        fps_effective = frame_idx / elapsed if elapsed > 0 else 0.0
        print(f"Done. Processed {frame_idx} frames in {elapsed:.2f}s ({fps_effective:.2f} FPS).")
        print(f"Output saved to: {output_path}")
        if computed_map is None:
            print("Note: mAP cannot be computed because no ground-truth annotations were provided.")

if __name__ == "__main__":
    main()