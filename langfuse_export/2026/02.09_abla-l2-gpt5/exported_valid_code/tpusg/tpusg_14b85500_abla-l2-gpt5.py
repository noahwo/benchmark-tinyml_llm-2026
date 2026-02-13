import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# ----------------------------- Configuration -----------------------------
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# ----------------------------- Utilities -----------------------------
def load_labels(path):
    """
    Loads labels from a label map file. Supports:
    - "index label" (space separated)
    - "index: label" or "index,label"
    - plain list (one label per line; index inferred)
    """
    labels = {}
    if not os.path.exists(path):
        print(f"Warning: Label file not found at {path}. Using empty labels.")
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try common formats
            idx = None
            name = None
            # Format: "0 label" or "0: label" or "0,label"
            for sep in [" ", ":", ",", "\t"]:
                parts = line.split(sep, 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    idx = int(parts[0].strip())
                    name = parts[1].strip()
                    break
            # Fallback: plain list
            if idx is None:
                idx = i
                name = line
            labels[idx] = name
    return labels


def make_interpreter(model_path, edgetpu_lib):
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(edgetpu_lib)]
        )
    except ValueError as e:
        raise RuntimeError(
            f"Failed to load the EdgeTPU delegate from {edgetpu_lib}. "
            f"Ensure the library exists and the model is compiled for EdgeTPU. Error: {e}"
        )
    interpreter.allocate_tensors()
    return interpreter


def preprocess(frame_bgr, input_shape, input_dtype):
    """
    - Convert BGR to RGB
    - Resize to model input size
    - Normalize if needed for float models
    - Add batch dimension
    """
    h, w = input_shape[1], input_shape[2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(input_dtype)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def get_output_tensors(interpreter):
    """
    Retrieve detection outputs from the interpreter in a model-agnostic way.
    Returns:
      boxes: (N, 4) in [ymin, xmin, ymax, xmax], normalized [0,1]
      classes: (N,) int
      scores: (N,) float32
      count: int
    """
    output_details = interpreter.get_output_details()
    tensors = [interpreter.get_tensor(od["index"]) for od in output_details]

    boxes = None
    classes = None
    scores = None
    count = None

    # Identify by shapes
    # Typically:
    # boxes: [1, N, 4] float32
    # classes: [1, N] float32/int
    # scores: [1, N] float32
    # count: [1] float32/int
    one_d_candidates = []
    for t in tensors:
        arr = np.squeeze(t)
        if arr.ndim == 2 and arr.shape[1] == 4:
            # Unlikely case, but handle
            boxes = arr
        elif arr.ndim == 2 and arr.shape[0] == 4:
            boxes = arr.T
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr[0]
        elif arr.ndim == 2:
            # Could be classes or scores
            one_d_candidates.append(arr[0] if arr.shape[0] == 1 else arr[:, 0])
        elif arr.ndim == 1 and arr.size > 1:
            one_d_candidates.append(arr)
        elif arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
            count = int(round(float(arr)))  # num_detections

    # Distinguish scores vs classes among 1D arrays
    # Scores are in [0,1], classes are integers (or floats close to ints)
    s_candidate = None
    c_candidate = None
    for cand in one_d_candidates:
        cmin, cmax = float(np.min(cand)), float(np.max(cand))
        # Heuristic: scores in [0,1], classes > 1 often
        if 0.0 <= cmin and cmax <= 1.0:
            s_candidate = cand
        else:
            c_candidate = cand

    if boxes is None:
        # Some models output boxes as [1, N, 4] float32
        for t in tensors:
            arr = np.array(t)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr[0]
                break

    if s_candidate is not None:
        scores = s_candidate.astype(np.float32)
    if c_candidate is not None:
        classes = c_candidate.astype(np.int32)

    # If count not provided, infer from scores or boxes
    if count is None:
        if scores is not None:
            count = int(scores.shape[0])
        elif boxes is not None:
            count = int(boxes.shape[0])
        elif classes is not None:
            count = int(classes.shape[0])
        else:
            count = 0

    # Ensure consistent lengths
    if boxes is not None and boxes.shape[0] > count:
        boxes = boxes[:count]
    if classes is not None and classes.shape[0] > count:
        classes = classes[:count]
    if scores is not None and scores.shape[0] > count:
        scores = scores[:count]

    # Final fallbacks
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((0,), dtype=np.int32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)

    return boxes, classes, scores, count


def draw_detections(frame, boxes, classes, scores, labels, threshold, map_value=None):
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        cls_id = int(classes[i]) if i < len(classes) else -1
        label = labels.get(cls_id, str(cls_id))
        ymin, xmin, ymax, xmax = boxes[i]
        # Convert normalized to pixel coords
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        # Color derived from class id for consistency
        color = tuple(int(c) for c in np.random.RandomState(cls_id).randint(0, 255, size=3))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        caption = f"{label}: {score*100:.1f}%"
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw mAP (proxy) on the frame if provided
    if map_value is not None:
        text = f"mAP: {map_value:.3f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


class MAPTracker:
    """
    A simple proxy metric to estimate mAP-like value without ground truth.
    For each class, we track:
      - TP: detections with score >= threshold
      - FP: detections with score < threshold
    AP per class is approximated as precision = TP / (TP + FP).
    mAP is the mean of per-class AP over observed classes.
    Note: This is NOT a replacement for true mAP which requires ground truth.
    """
    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)
        self.tp = {}   # class_id -> count
        self.fp = {}   # class_id -> count

    def update(self, classes, scores):
        for cls_id, sc in zip(classes, scores):
            cls = int(cls_id)
            if sc >= self.threshold:
                self.tp[cls] = self.tp.get(cls, 0) + 1
            else:
                self.fp[cls] = self.fp.get(cls, 0) + 1

    def compute_map(self):
        aps = []
        all_classes = set(list(self.tp.keys()) + list(self.fp.keys()))
        for cls in all_classes:
            tp = self.tp.get(cls, 0)
            fp = self.fp.get(cls, 0)
            denom = tp + fp
            if denom > 0:
                aps.append(tp / denom)
        if not aps:
            return 0.0
        return float(np.mean(aps))


def main():
    # Prepare output directory
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_LIB)
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]  # [1, height, width, channels]
    input_dtype = input_details[0]["dtype"]

    # Open video input
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    # Prepare video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    map_tracker = MAPTracker(threshold=CONFIDENCE_THRESHOLD)

    frame_count = 0
    inf_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            input_data = preprocess(frame, input_shape, input_dtype)

            # Set input tensor
            interpreter.set_tensor(input_index, input_data)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            inf_times.append(t1 - t0)

            # Postprocess outputs
            boxes, classes, scores, count = get_output_tensors(interpreter)

            # Update proxy mAP tracker
            if len(scores) > 0 and len(classes) == len(scores):
                map_tracker.update(classes, scores)

            # Compute current mAP (proxy)
            current_map = map_tracker.compute_map()

            # Draw detections and mAP
            draw_detections(frame, boxes, classes, scores, labels, CONFIDENCE_THRESHOLD, map_value=current_map)

            # Write frame
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    # Report
    avg_inf_ms = (np.mean(inf_times) * 1000.0) if inf_times else 0.0
    final_map = map_tracker.compute_map()
    print("Processing complete.")
    print(f"Frames processed: {frame_count}")
    print(f"Average inference time: {avg_inf_ms:.2f} ms")
    print(f"Saved output video to: {OUTPUT_PATH}")
    print(f"mAP (proxy without ground truth): {final_map:.4f}")


if __name__ == "__main__":
    main()