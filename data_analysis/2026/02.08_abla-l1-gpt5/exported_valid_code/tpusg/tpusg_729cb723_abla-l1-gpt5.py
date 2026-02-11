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
# Utility Functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print("Label file not found:", path)
        return labels
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " " in line:
                # Try "index label" format
                parts = line.split(maxsplit=1)
                if parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                else:
                    # Fallback: no explicit index; assign sequentially
                    labels[len(labels)] = line
            else:
                labels[len(labels)] = line
    return labels

def make_interpreter(model_path):
    delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(delegate_path)]
        )
        return interpreter
    except Exception as e:
        print("Failed to load EdgeTPU delegate at", delegate_path)
        raise e

def get_input_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    shape = input_details["shape"]
    dtype = input_details["dtype"]
    # Assume 4D input tensor
    if len(shape) != 4:
        raise RuntimeError("Unexpected input tensor shape: {}".format(shape))
    # Detect layout
    # Common: NHWC => shape: [1, height, width, 3]
    # Less common: NCHW => shape: [1, 3, height, width]
    if shape[3] == 3:
        layout = "NHWC"
        height, width = int(shape[1]), int(shape[2])
    elif shape[1] == 3:
        layout = "NCHW"
        height, width = int(shape[2]), int(shape[3])
    else:
        raise RuntimeError("Cannot infer input layout from shape: {}".format(shape))
    return {
        "index": input_details["index"],
        "shape": shape,
        "dtype": dtype,
        "layout": layout,
        "height": height,
        "width": width
    }

def set_input_tensor(interpreter, frame_bgr, inp):
    # Convert BGR to RGB as most TFLite models expect RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (inp["width"], inp["height"]))
    if inp["dtype"] == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(inp["dtype"])
    if inp["layout"] == "NHWC":
        input_data = np.expand_dims(input_data, axis=0)  # [1, H, W, 3]
    else:
        # NCHW
        input_data = np.transpose(input_data, (2, 0, 1))  # [3, H, W]
        input_data = np.expand_dims(input_data, axis=0)   # [1, 3, H, W]
    interpreter.set_tensor(inp["index"], input_data)

def _extract_outputs_by_name(interpreter, output_details):
    boxes = None
    classes = None
    scores = None
    count = None
    for d in output_details:
        name = d.get("name", "").lower()
        data = interpreter.get_tensor(d["index"])
        # Most models return batch dimension at 0; squeeze that if present
        if data.ndim >= 2 and data.shape[0] == 1:
            data_squeezed = np.squeeze(data, axis=0)
        else:
            data_squeezed = data
        if "box" in name:
            boxes = data_squeezed
        elif "score" in name:
            scores = data_squeezed
        elif "class" in name:
            classes = data_squeezed
        elif "count" in name or "num" in name:
            try:
                count = int(np.array(data).reshape(-1)[0])
            except Exception:
                count = None
    return boxes, classes, scores, count

def _fallback_extract_outputs(interpreter, output_details):
    outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
    # Squeeze batch if present
    outs = []
    for arr in outputs:
        if arr.ndim >= 2 and arr.shape[0] == 1:
            outs.append(np.squeeze(arr, axis=0))
        else:
            outs.append(arr)
    boxes = None
    classes = None
    scores = None
    count = None
    # Identify boxes (shape (..., 4))
    for arr in outs:
        if arr.ndim >= 2 and arr.shape[-1] == 4:
            boxes = arr
            break
    # Identify count (scalar)
    for arr in outs:
        if arr.size == 1:
            try:
                count = int(arr.reshape(-1)[0])
                break
            except Exception:
                pass
    # Remaining vectors -> classes and scores
    candidates = [a for a in outs if not (a is boxes or (a.size == 1))]
    # Heuristic: scores usually float and between 0 and 1
    score_cands = []
    class_cands = []
    for a in candidates:
        if a.dtype in (np.float32, np.float64) and np.all((a >= 0) & (a <= 1)):
            score_cands.append(a)
        else:
            class_cands.append(a)
    scores = score_cands[0] if score_cands else (candidates[0] if candidates else None)
    classes = class_cands[0] if class_cands else (candidates[1] if len(candidates) > 1 else None)
    return boxes, classes, scores, count

def get_detections(interpreter, threshold, frame_w, frame_h):
    output_details = interpreter.get_output_details()
    boxes, classes, scores, count = _extract_outputs_by_name(interpreter, output_details)
    if boxes is None or classes is None or scores is None:
        boxes, classes, scores, count = _fallback_extract_outputs(interpreter, output_details)

    if boxes is None or classes is None or scores is None:
        return []

    # Ensure 1D arrays for classes, scores
    classes = np.array(classes).reshape(-1)
    scores = np.array(scores).reshape(-1)
    if boxes.ndim == 3:
        # [N, num, 4] -> assume single batch squeezed incorrectly; pick first
        boxes = boxes[0]
    boxes = np.array(boxes).reshape(-1, 4)

    if count is None:
        n = min(len(boxes), len(classes), len(scores))
    else:
        n = min(int(count), len(boxes), len(classes), len(scores))

    detections = []
    for i in range(n):
        score = float(scores[i])
        if score < threshold:
            continue
        # Boxes are [ymin, xmin, ymax, xmax] normalized [0,1] typically
        y_min, x_min, y_max, x_max = boxes[i]
        x_min_px = max(0, min(int(x_min * frame_w), frame_w - 1))
        x_max_px = max(0, min(int(x_max * frame_w), frame_w - 1))
        y_min_px = max(0, min(int(y_min * frame_h), frame_h - 1))
        y_max_px = max(0, min(int(y_max * frame_h), frame_h - 1))
        class_id = int(classes[i]) if not np.isnan(classes[i]) else -1
        detections.append({
            "bbox": (x_min_px, y_min_px, x_max_px, y_max_px),
            "class_id": class_id,
            "score": score
        })
    return detections

def draw_detections(frame_bgr, detections, labels):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        score = det["score"]
        label = labels.get(cls_id, str(cls_id))
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = "{}: {:.2f}".format(label, score)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - 5)
        cv2.rectangle(frame_bgr, (x1, y_text - th - baseline), (x1 + tw, y_text + baseline), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

class MAPTracker:
    def __init__(self):
        # Accumulate per-class confidences as a proxy for AP (requires GT otherwise)
        self.class_sum = {}
        self.class_count = {}

    def update(self, detections):
        for det in detections:
            cls = det["class_id"]
            sc = float(det["score"])
            self.class_sum[cls] = self.class_sum.get(cls, 0.0) + sc
            self.class_count[cls] = self.class_count.get(cls, 0) + 1

    def compute_ap_per_class(self):
        ap = {}
        for cls in self.class_sum:
            cnt = self.class_count.get(cls, 0)
            if cnt > 0:
                ap[cls] = self.class_sum[cls] / float(cnt)
        return ap

    def compute_map(self):
        ap = self.compute_ap_per_class()
        if not ap:
            return 0.0
        return float(np.mean(list(ap.values())))

def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Initialize interpreter with EdgeTPU
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    inp = get_input_details(interpreter)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Failed to open input video:", input_path)
        return

    # Get video properties
    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # default fallback

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (in_width, in_height))
    if not writer.isOpened():
        print("Failed to open output video for writing:", output_path)
        cap.release()
        return

    # Stats
    map_tracker = MAPTracker()
    frame_index = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        # Preprocess and set input tensor
        set_input_tensor(interpreter, frame, inp)

        # Inference
        interpreter.invoke()

        # Get detections
        detections = get_detections(interpreter, confidence_threshold, frame.shape[1], frame.shape[0])

        # Update mAP tracker (proxy using mean confidence per class)
        map_tracker.update(detections)
        current_map = map_tracker.compute_map()

        # Draw detections and mAP
        draw_detections(frame, detections, labels)
        map_text = "mAP: {:.3f}".format(current_map)
        cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Write frame to output
        writer.write(frame)

    t1 = time.time()
    elapsed = t1 - t0
    total_frames = frame_index
    overall_fps = (total_frames / elapsed) if elapsed > 0 else 0.0

    # Cleanup
    cap.release()
    writer.release()

    # Final report
    final_map = map_tracker.compute_map()
    print("Processing complete.")
    print("Frames processed:", total_frames)
    print("Elapsed time (s): {:.2f}".format(elapsed))
    print("Throughput (FPS): {:.2f}".format(overall_fps))
    print("Estimated mAP (proxy using mean confidence per class): {:.3f}".format(final_map))
    print("Output saved to:", output_path)

if __name__ == "__main__":
    main()