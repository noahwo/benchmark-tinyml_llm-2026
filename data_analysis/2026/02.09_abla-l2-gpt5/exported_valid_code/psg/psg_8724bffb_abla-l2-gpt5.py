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
INPUT_PATH = "data/object_detection/sheeps.mp4"   # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Save video with rectangles, labels, and mAP
CONFIDENCE_THRESHOLD = 0.5


def load_labels(label_path):
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                labels[idx] = name
    return labels


def make_dirs_for_file(file_path):
    out_dir = os.path.dirname(os.path.abspath(file_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def preprocess_frame(frame_bgr, input_w, input_h, input_dtype):
    # Convert BGR (OpenCV) to RGB (most models expect RGB)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    else:
        # Assume uint8 quantized model
        inp = resized.astype(np.uint8)
    # Add batch dimension
    inp = np.expand_dims(inp, axis=0)
    return inp


def parse_outputs(interpreter):
    """
    Robustly extract detection outputs: boxes, classes, scores, num_detections
    Works with common SSD Mobilenet TFLite postprocess outputs.
    Returns:
        boxes: (N,4) in [ymin, xmin, ymax, xmax] normalized
        classes: (N,) float/int class ids
        scores: (N,) float scores
        num_detections: int
    """
    output_details = interpreter.get_output_details()
    # Fetch all outputs once
    tensors = []
    for od in output_details:
        data = interpreter.get_tensor(od["index"])
        tensors.append((od, data))

    boxes = None
    classes = None
    scores = None
    num_det = None

    # First try by name
    for od, data in tensors:
        name = od.get("name", "").lower()
        if "box" in name:
            boxes = np.squeeze(data, axis=0)
        elif "score" in name:
            scores = np.squeeze(data, axis=0)
        elif "class" in name:
            classes = np.squeeze(data, axis=0)
        elif "num" in name:
            num_det = int(np.squeeze(data))

    # Fallback by shape/content if name-based failed
    if boxes is None or scores is None or classes is None or num_det is None:
        # Identify boxes: shape (1, N, 4)
        for od, data in tensors:
            if data.ndim == 3 and data.shape[-1] == 4:
                boxes = data[0]

        cand = []
        single = []
        for od, data in tensors:
            if od.get("name", "").lower().find("box") >= 0:
                continue
            if data.ndim == 2 and data.shape[0] == 1:
                cand.append(np.squeeze(data, axis=0))
            if data.size == 1:
                single.append(int(np.squeeze(data)))

        # Among cand, identify scores as the one largely in [0,1]
        if len(cand) >= 2:
            conf_idx = None
            for i, arr in enumerate(cand):
                if arr.dtype.kind == "f":
                    # fraction of values in [0,1]
                    frac_in_01 = np.mean((arr >= 0.0) & (arr <= 1.0))
                    if frac_in_01 > 0.8:
                        conf_idx = i
                        break
            if conf_idx is not None:
                scores = cand[conf_idx]
                classes = cand[1 - conf_idx]
            else:
                # Default assign
                scores = cand[0]
                classes = cand[1]
        if num_det is None:
            num_det = single[0] if single else (scores.shape[0] if scores is not None else 0)

    # Ensure proper shapes/types
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.float32)

    # Clip num_det to available arrays
    n = int(num_det) if isinstance(num_det, (int, np.integer)) else int(np.squeeze(num_det))
    n = min(n, boxes.shape[0], scores.shape[0], classes.shape[0])
    return boxes[:n], classes[:n], scores[:n], n


class MAPAggregator:
    """
    Simple aggregator to compute a running mAP-like metric across frames.
    Here, AP per class is approximated as the mean confidence of detections
    above threshold for that class. mAP is the mean of per-class APs.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.class_scores = {}  # class_id -> list of scores

    def update(self, classes, scores):
        for c, s in zip(classes, scores):
            if s >= self.threshold:
                cid = int(c)
                if cid not in self.class_scores:
                    self.class_scores[cid] = []
                self.class_scores[cid].append(float(s))

    def compute_map(self):
        if not self.class_scores:
            return 0.0
        aps = []
        for _, vals in self.class_scores.items():
            if len(vals) == 0:
                continue
            aps.append(float(np.mean(vals)))
        if not aps:
            return 0.0
        return float(np.mean(aps))


def draw_detections(frame_bgr, detections, labels, map_value):
    h, w = frame_bgr.shape[:2]
    boxes, classes, scores = detections

    for box, cls_id_f, score in zip(boxes, classes, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = box
        left = max(0, min(w - 1, int(xmin * w)))
        right = max(0, min(w - 1, int(xmax * w)))
        top = max(0, min(h - 1, int(ymin * h)))
        bottom = max(0, min(h - 1, int(ymax * h)))

        cls_id = int(cls_id_f)
        # Safe label fetch
        label_text = labels.get(cls_id, f"id:{cls_id}")
        color = (0, 255, 0)  # green boxes
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

        text = f"{label_text} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (left, top - th - baseline - 3), (left + tw + 4, top), color, -1)
        cv2.putText(frame_bgr, text, (left + 2, top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay mAP value
    map_text = f"mAP: {map_value:.3f}"
    (tw, th), baseline = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (10, 10), (10 + tw + 10, 10 + th + baseline + 10), (255, 255, 255), -1)
    cv2.putText(frame_bgr, map_text, (15, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    # 1. Setup: Interpreter, labels, video I/O
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

    labels = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    input_idx = input_details["index"]
    in_dtype = input_details["dtype"]
    # Expect shape [1, height, width, 3]
    in_h = int(input_details["shape"][1])
    in_w = int(input_details["shape"][2])

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    make_dirs_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {OUTPUT_PATH}")

    map_agg = MAPAggregator(CONFIDENCE_THRESHOLD)

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 2. Preprocessing
            inp = preprocess_frame(frame, in_w, in_h, in_dtype)

            # 3. Inference
            interpreter.set_tensor(input_idx, inp)
            interpreter.invoke()

            boxes, classes, scores, n = parse_outputs(interpreter)

            # Update mAP aggregator with current detections
            map_agg.update(classes, scores)
            current_map = map_agg.compute_map()

            # 4. Output handling: draw and write frame
            draw_detections(frame, (boxes, classes, scores), labels, current_map)
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    final_map = map_agg.compute_map()
    print(f"Processed {frame_count} frames in {elapsed:.2f}s (avg {avg_fps:.2f} FPS)")
    print(f"Final mAP: {final_map:.4f}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()