import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration parameters
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Output with rectangles, labels, and mAP text overlay
CONFIDENCE_THRESHOLD = 0.5

# =========================
# Utilities
# =========================
def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line)
    return labels

def get_label_text(labels, class_id):
    # Handle common label map conventions
    ci = int(class_id)
    if 0 <= ci < len(labels):
        # Typical label files for TFLite SSD start with "???" at index 0 (background)
        # and classes are 0-based. This handles that case directly.
        txt = labels[ci]
        if txt == "???":  # if background appears, fall back to id text
            return f"id:{ci}"
        return txt
    # Fallback for 1-based label maps
    if 1 <= ci <= len(labels):
        return labels[ci - 1]
    return f"id:{ci}"

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def preprocess_frame_bgr_to_input(frame_bgr, input_shape, input_dtype, input_quant):
    # input_shape: [1, h, w, 3]
    _, in_h, in_w, _ = input_shape
    # Convert BGR (OpenCV) to RGB and resize to model input size
    resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        # Normalize to [0,1] for typical SSD Mobilenet v1 float models
        tensor = rgb.astype(np.float32) / 255.0
    else:
        # uint8 path
        # For quantized models, TFLite typically expects uint8 [0,255].
        tensor = rgb.astype(np.uint8)

        # If you need explicit quantization to uint8 (rarely necessary here),
        # you could apply: (float_tensor / scale + zero_point), but most models
        # expect raw uint8 image; keep it simple and correct for common cases.
        # scale, zero_point = input_quant
        # if scale and scale > 0:
        #     tensor = np.clip(np.round(tensor / scale + zero_point), 0, 255).astype(np.uint8)

    # Add batch dimension
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

def parse_tflite_outputs(interpreter):
    # Retrieve all output tensors and identify boxes, classes, scores, and num_detections
    outputs = []
    out_details = interpreter.get_output_details()
    for od in out_details:
        arr = interpreter.get_tensor(od['index'])
        outputs.append((od, arr))

    boxes = None
    classes = None
    scores = None
    num_detections = None

    # First pass by name hints if available
    for od, arr in outputs:
        name = od.get('name', '')
        shp = arr.shape
        if 'boxes' in name and arr.ndim == 3 and shp[-1] == 4:
            boxes = arr
        elif 'classes' in name and arr.ndim == 2:
            classes = arr
        elif 'scores' in name and arr.ndim == 2:
            scores = arr
        elif 'num_detections' in name and arr.size == 1:
            num_detections = int(np.squeeze(arr).astype(np.int32))

    # Fallback by shapes and value ranges
    if boxes is None or classes is None or scores is None or num_detections is None:
        # Identify boxes: [1, N, 4]
        for _, arr in outputs:
            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 4:
                boxes = arr
        # Identify num_detections: scalar [1] or shape () after squeeze
        for _, arr in outputs:
            if arr.size == 1:
                num_detections = int(np.squeeze(arr).astype(np.int32))
        # Identify classes and scores among [1, N]
        two_d = [arr for _, arr in outputs if arr.ndim == 2 and arr.shape[0] == 1]
        # If we have exactly two candidates
        if len(two_d) >= 2:
            # Scores are in [0,1], classes are class indices (float)
            # Choose the one with values mostly between 0 and 1 as scores
            cand_a, cand_b = two_d[0], two_d[1]
            a_max, b_max = float(np.max(cand_a)), float(np.max(cand_b))
            a_min, b_min = float(np.min(cand_a)), float(np.min(cand_b))
            def likely_scores(lo, hi):
                return (lo >= 0.0) and (hi <= 1.0 + 1e-4)
            if likely_scores(a_min, a_max) and not likely_scores(b_min, b_max):
                scores, classes = cand_a, cand_b
            elif likely_scores(b_min, b_max) and not likely_scores(a_min, a_max):
                scores, classes = cand_b, cand_a
            else:
                # Ambiguous; assume first is scores for typical TFLite SSD exports
                scores, classes = cand_a, cand_b

    # Final sanity
    if boxes is None or classes is None or scores is None:
        raise RuntimeError("Unable to parse TFLite SSD outputs (boxes/classes/scores not found).")

    if num_detections is None:
        # Some models omit num_detections; infer from array length
        num_detections = boxes.shape[1]

    # Squeeze batch dimension
    boxes = np.squeeze(boxes, axis=0)
    classes = np.squeeze(classes, axis=0)
    scores = np.squeeze(scores, axis=0)

    # Truncate to num_detections if needed
    boxes = boxes[:num_detections]
    classes = classes[:num_detections]
    scores = scores[:num_detections]

    return boxes, classes, scores, num_detections

def draw_detections_on_frame(frame_bgr, detections, map_text):
    # detections: list of dicts with keys: bbox (ymin, xmin, ymax, xmax) normalized,
    # score, class_id, label
    h, w = frame_bgr.shape[:2]

    # Draw mAP text (top-left)
    cv2.rectangle(frame_bgr, (5, 5), (260, 35), (0, 0, 0), thickness=-1)
    cv2.putText(frame_bgr, map_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    for det in detections:
        ymin, xmin, ymax, xmax = det["bbox"]
        # Scale to pixel coordinates
        x1 = int(max(0, min(w - 1, xmin * w)))
        y1 = int(max(0, min(h - 1, ymin * h)))
        x2 = int(max(0, min(w - 1, xmax * w)))
        y2 = int(max(0, min(h - 1, ymax * h)))

        # Box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        label = det["label"]
        score = det["score"]
        text = f"{label}: {score:.2f}"

        # Text background
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(frame_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + 6), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (x1 + 2, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr

def calculate_map_placeholder():
    # Proper mAP requires ground-truth annotations.
    # Since no ground-truth is provided, we cannot compute a valid mAP.
    # Return a display string that clearly indicates unavailability.
    return "mAP: N/A (no GT)"

# =========================
# Main pipeline
# =========================
def main():
    # Setup and interpreter
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    if not input_details:
        raise RuntimeError("Interpreter has no input tensors.")
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # [1, h, w, 3]
    input_dtype = input_details[0]['dtype']
    input_quant = input_details[0].get('quantization', (0.0, 0))

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 1e-3:
        fps_in = 30.0  # fallback

    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # Placeholder mAP text (no ground-truth available in configuration)
    map_text = calculate_map_placeholder()

    # Processing loop
    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # Preprocess
        input_tensor = preprocess_frame_bgr_to_input(frame_bgr, input_shape, input_dtype, input_quant)

        # Set input tensor
        interpreter.set_tensor(input_index, input_tensor)

        # Inference
        interpreter.invoke()

        # Parse outputs
        boxes, classes, scores, num = parse_tflite_outputs(interpreter)

        # Collect detections above threshold
        detections = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            class_id = int(classes[i])
            label = get_label_text(labels, class_id)
            ymin, xmin, ymax, xmax = boxes[i].tolist()  # normalized coordinates
            det = {
                "bbox": (ymin, xmin, ymax, xmax),
                "score": score,
                "class_id": class_id,
                "label": label
            }
            detections.append(det)

        # Draw detections and mAP text
        annotated = draw_detections_on_frame(frame_bgr, detections, map_text)

        # Write frame
        writer.write(annotated)

    # Cleanup
    cap.release()
    writer.release()
    elapsed = time.time() - t0
    if frame_count > 0:
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count / max(elapsed,1e-6):.2f} FPS).")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()