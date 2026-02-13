import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# =========================
# Utility Functions
# =========================
def load_labels(path):
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support both "id label" and plain "label" formats
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)
    return labels

def ensure_dir_for_file(filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def resolve_output_indices(output_details):
    # Attempt to map output tensors to: boxes, classes, scores, num
    idx = {"boxes": None, "classes": None, "scores": None, "count": None}
    for i, d in enumerate(output_details):
        shape = d.get("shape", [])
        dtype = d.get("dtype", None)
        name = d.get("name", "").lower()
        # Heuristics based on common TFLite detection models
        if len(shape) == 3 and shape[-1] == 4 and dtype == np.float32:
            idx["boxes"] = i
        elif len(shape) == 2 and dtype in (np.float32, np.uint8):  # scores or classes usually (1,N)
            # Try to disambiguate via name
            if "score" in name or "scores" in name:
                idx["scores"] = i
            elif "class" in name or "classes" in name:
                idx["classes"] = i
        elif len(shape) == 1 and shape[0] == 1:
            # Could be num_detections or something else
            if "num" in name or "count" in name or "detection" in name:
                idx["count"] = i

    # If names unavailable, fall back to typical 4-output ordering
    if any(v is None for v in idx.values()) and len(output_details) >= 4:
        # Common order: boxes, classes, scores, num_detections
        # We'll attempt to detect by shapes:
        candidates = list(range(len(output_details)))
        # boxes: (1,N,4)
        for i, d in enumerate(output_details):
            s = d.get("shape", [])
            if len(s) == 3 and s[-1] == 4 and idx["boxes"] is None:
                idx["boxes"] = i
                if i in candidates: candidates.remove(i)
        # scores/classes: (1,N)
        remaining_1n = [i for i in candidates if len(output_details[i].get("shape", [])) == 2]
        # Assign with heuristics by name if any
        for i in remaining_1n:
            n = output_details[i].get("name","").lower()
            if "score" in n and idx["scores"] is None:
                idx["scores"] = i
            if "class" in n and idx["classes"] is None:
                idx["classes"] = i
        # If still missing, just assign arbitrarily from remaining
        remaining_1n = [i for i in candidates if len(output_details[i].get("shape", [])) == 2]
        if idx["scores"] is None and remaining_1n:
            idx["scores"] = remaining_1n.pop(0)
        if idx["classes"] is None and remaining_1n:
            idx["classes"] = remaining_1n.pop(0)
        # num detections: (1,)
        for i in candidates:
            s = output_details[i].get("shape", [])
            if len(s) == 1 and s[0] == 1 and idx["count"] is None:
                idx["count"] = i

    return idx

def preprocess_frame(frame_bgr, input_size, input_dtype):
    ih, iw = input_size
    # Resize and convert color
    resized = cv2.resize(frame_bgr, (iw, ih))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Prepare tensor
    if input_dtype == np.float32:
        tensor = (rgb.astype(np.float32) / 255.0)
    else:
        tensor = rgb.astype(np.uint8)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

def draw_detections(frame, boxes, classes, scores, labels, threshold):
    h, w = frame.shape[:2]
    count = 0
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        count += 1
        cls_id = int(classes[i]) if i < len(classes) else -1
        label_text = labels[cls_id] if (0 <= cls_id < len(labels)) else f"id {cls_id}"
        # boxes are [ymin, xmin, ymax, xmax] normalized
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(max(0, xmin * w)), int(max(0, ymin * h))
        x2, y2 = int(min(w - 1, xmax * w)), int(min(h - 1, ymax * h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        caption = f"{label_text}: {score:.2f}"
        cv2.putText(frame, caption, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)
    return count

# =========================
# Main Pipeline
# =========================
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input video not found: {INPUT_PATH}")
        return

    ensure_dir_for_file(OUTPUT_PATH)

    # Load labels
    labels = load_labels(LABEL_PATH)
    if not labels:
        print(f"Warning: No labels found at {LABEL_PATH}. Proceeding without label names.")

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=max(1, os.cpu_count() or 1))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine input tensor properties
    in_det = input_details[0]
    in_shape = in_det.get("shape", [1, 300, 300, 3])
    in_dtype = in_det.get("dtype", np.uint8)
    input_h, input_w = int(in_shape[1]), int(in_shape[2])

    # Map output indices
    out_idx = resolve_output_indices(output_details)
    if any(out_idx[k] is None for k in ["boxes", "classes", "scores", "count"]):
        print("Error: Could not resolve output tensor indices for detection model.")
        return

    # Video I/O setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {INPUT_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Processing loop
    avg_infer_ms = 0.0
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocessing
            input_tensor = preprocess_frame(frame, (input_h, input_w), in_dtype)

            # Inference
            interpreter.set_tensor(in_det["index"], input_tensor)
            t_infer_start = time.time()
            interpreter.invoke()
            t_infer = (time.time() - t_infer_start) * 1000.0  # ms
            frame_count += 1
            avg_infer_ms += (t_infer - avg_infer_ms) / frame_count

            # Retrieve outputs
            boxes = interpreter.get_tensor(output_details[out_idx["boxes"]]["index"])
            classes = interpreter.get_tensor(output_details[out_idx["classes"]]["index"])
            scores = interpreter.get_tensor(output_details[out_idx["scores"]]["index"])
            num = interpreter.get_tensor(output_details[out_idx["count"]]["index"])

            # Squeeze to expected shapes
            boxes = np.squeeze(boxes, axis=0) if boxes.ndim == 3 else boxes
            classes = np.squeeze(classes, axis=0) if classes.ndim == 2 else classes
            scores = np.squeeze(scores, axis=0) if scores.ndim == 2 else scores
            # num detections (may be float)
            num_detections = int(np.squeeze(num).astype(np.int32))

            # Clip arrays to num_detections if necessary
            if boxes.shape[0] > num_detections:
                boxes = boxes[:num_detections]
            if classes.shape[0] > num_detections:
                classes = classes[:num_detections]
            if scores.shape[0] > num_detections:
                scores = scores[:num_detections]

            # Output handling: draw detections
            det_count = draw_detections(frame, boxes, classes, scores, labels, CONFIDENCE_THRESHOLD)

            # Overlay performance info
            fps_runtime = frame_count / max(1e-6, (time.time() - t0))
            cv2.putText(frame, f"Infer: {t_infer:.1f} ms (avg {avg_infer_ms:.1f} ms) | FPS: {fps_runtime:.1f} | Dets: {det_count}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 220, 220), 2, cv2.LINE_AA)

            # Write frame
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"Processing complete. Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()