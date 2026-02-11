import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# ==============
# UTIL FUNCTIONS
# ==============
def load_labels(label_path):
    labels = {}
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                labels[idx] = name
    return labels

def class_color(cid):
    # Simple deterministic color generator based on class id
    np.random.seed(cid + 7)
    color = np.random.randint(50, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def preprocess_frame(frame, input_size, input_dtype):
    # Convert BGR to RGB and resize to model input
    h_in, w_in = input_size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def find_output_tensors_details(output_details):
    """
    Identify indices for boxes, classes, scores, num_detections by inspecting tensor shapes.
    Expected:
    - boxes: shape [1, N, 4]
    - classes: shape [1, N]
    - scores: shape [1, N]
    - num_detections: shape [1] or []
    """
    boxes_i = classes_i = scores_i = num_i = None
    for i, d in enumerate(output_details):
        shape = d.get("shape", [])
        if len(shape) == 3 and shape[-1] == 4:
            boxes_i = i
        elif len(shape) == 2 and shape[0] == 1:
            # Could be classes or scores; delay assignment
            # We'll assign later based on dtype if available (both float), so we fallback by first/second encounter
            if scores_i is None:
                scores_i = i
            elif classes_i is None:
                classes_i = i
        elif len(shape) in (0, 1):  # num_detections might be [1] or []
            num_i = i

    # If classes and scores might be swapped, try to disambiguate by name if present
    # Typical names contain 'scores' or 'classes'
    for i, d in enumerate(output_details):
        name = d.get("name", "").lower()
        if "score" in name:
            scores_i = i
        if "class" in name:
            classes_i = i
        if "num" in name and "detection" in name:
            num_i = i

    return boxes_i, classes_i, scores_i, num_i

def draw_detections(frame, boxes, classes, scores, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        cid = int(classes[i])
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(max(0.0, xmin) * w), int(max(0.0, ymin) * h)
        x2, y2 = int(min(1.0, xmax) * w), int(min(1.0, ymax) * h)
        color = class_color(cid)
        label_text = labels.get(cid, f"id:{cid}")
        caption = f"{label_text} {score:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw filled background for text
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame

def main():
    # 1) SETUP
    ensure_dir(OUTPUT_PATH)
    labels = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        raise RuntimeError("No input tensors found in the TFLite model.")

    input_index = input_details[0]["index"]
    # Expect NHWC
    _, in_h, in_w, in_c = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Prepare outputs mapping
    b_i, c_i, s_i, n_i = find_output_tensors_details(output_details)
    if None in (b_i, c_i, s_i):
        # Fall back to typical ordering if detection fails
        if len(output_details) >= 3:
            b_i, c_i, s_i = 0, 1, 2
        if len(output_details) >= 4 and n_i is None:
            n_i = 3

    # Video I/O
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # 2) PREPROCESSING + 3) INFERENCE + 4) OUTPUT HANDLING
    frame_count = 0
    t_start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_data = preprocess_frame(frame, (in_h, in_w), input_dtype)
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()

            # Retrieve outputs
            boxes = interpreter.get_tensor(output_details[b_i]["index"]) if b_i is not None else None
            classes = interpreter.get_tensor(output_details[c_i]["index"]) if c_i is not None else None
            scores = interpreter.get_tensor(output_details[s_i]["index"]) if s_i is not None else None

            # Squeeze batch dimension if present
            if boxes is not None and boxes.ndim == 3:
                boxes = boxes[0]
            if classes is not None and classes.ndim == 2:
                classes = classes[0]
            if scores is not None and scores.ndim == 2:
                scores = scores[0]

            # Draw detections
            if boxes is not None and classes is not None and scores is not None:
                frame = draw_detections(frame, boxes, classes, scores, labels, CONFIDENCE_THRESHOLD)

            out.write(frame)
            frame_count += 1

    finally:
        cap.release()
        out.release()
        t_elapsed = max(1e-6, time.time() - t_start)
        avg_fps = frame_count / t_elapsed
        print(f"Processed {frame_count} frames in {t_elapsed:.2f}s (avg {avg_fps:.2f} FPS).")
        print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()