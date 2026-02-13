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


def load_labels(path):
    labels = {}
    try:
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Support "id label", "id: label", or plain "label"
                if ":" in line:
                    parts = [p.strip() for p in line.split(":", 1)]
                    if len(parts) == 2 and parts[0].isdigit():
                        labels[int(parts[0])] = parts[1]
                        continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                else:
                    labels[idx] = line
    except Exception as e:
        print(f"Warning: Failed to load labels from {path}: {e}")
    return labels


def make_interpreter(model_path, delegate_path):
    return Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(delegate_path)]
    )


def preprocess(frame_bgr, input_size, input_dtype):
    # Convert BGR (OpenCV) to RGB as most TFLite detection models expect RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_size[1], input_size[0]))
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def postprocess(frame_shape, boxes, classes, scores, num, conf_thres):
    h, w = frame_shape[:2]
    detections = []
    count = int(num)
    for i in range(count):
        score = float(scores[i])
        if score < conf_thres:
            continue
        cls_id = int(classes[i])
        # boxes are typically in [ymin, xmin, ymax, xmax] normalized coordinates
        y_min, x_min, y_max, x_max = boxes[i]
        x0 = max(0, min(w - 1, int(x_min * w)))
        y0 = max(0, min(h - 1, int(y_min * h)))
        x1 = max(0, min(w - 1, int(x_max * w)))
        y1 = max(0, min(h - 1, int(y_max * h)))
        # Ensure proper rectangle coordinates
        x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
        y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
        detections.append((x0, y0, x1, y1, cls_id, score))
    return detections


def draw_detections(frame, detections, labels, map_value=None):
    for (x0, y0, x1, y1, cls_id, score) in detections:
        color = (0, 255, 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        # Try both 0-based and 1-based indexing for labels
        label_text = labels.get(cls_id, labels.get(cls_id + 1, str(cls_id)))
        text = f"{label_text}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x0, y0 - th - baseline), (x0 + tw, y0), color, -1)
        cv2.putText(frame, text, (x0, y0 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if map_value is not None:
        map_text = f"mAP (proxy): {map_value:.3f}"
        cv2.putText(frame, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 255), 2, cv2.LINE_AA)


# Heuristic, proxy mAP computation without ground-truth:
# - For each class and frame, the highest-scoring detection is treated as TP, all other detections of that class in that frame are FP.
# - AP is computed across frames based on scores, integrating the precision-recall curve where recall is TP / (number of frames with at least one detection for that class).
# This does not reflect true mAP without ground-truth, but serves as a proxy metric for demonstration.
def update_ap_data_for_frame(detections, ap_data, num_pos_by_class):
    # Sort detections in current frame by score descending per class
    per_class = {}
    for det in detections:
        x0, y0, x1, y1, cls_id, score = det
        per_class.setdefault(cls_id, []).append(score)
    # For each class, mark the highest-scoring detection as TP, others as FP
    for cls_id, scores in per_class.items():
        scores_sorted = sorted(scores, reverse=True)
        for j, s in enumerate(scores_sorted):
            tp_flag = 1 if j == 0 else 0
            ap_data.setdefault(cls_id, []).append((float(s), tp_flag))
        # Count one positive "instance" for this frame for that class (proxy)
        num_pos_by_class[cls_id] = num_pos_by_class.get(cls_id, 0) + 1


def compute_ap(scores_and_tp, num_pos):
    if num_pos <= 0 or len(scores_and_tp) == 0:
        return None
    # Sort by score descending
    arr = sorted(scores_and_tp, key=lambda x: x[0], reverse=True)
    tps = np.array([tp for _, tp in arr], dtype=np.float32)
    fps = 1.0 - tps
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
    recalls = cum_tp / (num_pos + 1e-8)

    # Precision envelope (monotonic decreasing)
    mprec = precisions.copy()
    for i in range(len(mprec) - 2, -1, -1):
        if mprec[i] < mprec[i + 1]:
            mprec[i] = mprec[i + 1]

    # Integrate area under precision-recall curve
    ap = 0.0
    prev_recall = 0.0
    for i in range(len(recalls)):
        if i == 0 or recalls[i] != recalls[i - 1]:
            ap += (recalls[i] - prev_recall) * mprec[i]
            prev_recall = recalls[i]
    # Bound AP to [0,1]
    ap = float(max(0.0, min(1.0, ap)))
    return ap


def compute_map(ap_data, num_pos_by_class):
    aps = []
    for cls_id, data in ap_data.items():
        ap = compute_ap(data, num_pos_by_class.get(cls_id, 0))
        if ap is not None:
            aps.append(ap)
    if len(aps) == 0:
        return None
    return float(np.mean(aps))


def main():
    # 1) Setup: Interpreter, labels, video IO
    labels = load_labels(label_path)
    interpreter = make_interpreter(model_path, edgetpu_lib)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    in_h, in_w = input_details[0]["shape"][1], input_details[0]["shape"][2]
    input_dtype = input_details[0]["dtype"]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Failed to open input video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: Failed to open video writer for: {output_path}")
        cap.release()
        return

    # For proxy mAP calculation
    ap_data = {}  # class_id -> list of (score, tp_flag)
    num_pos_by_class = {}  # class_id -> count of frames with at least one detection

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 2) Preprocessing
            input_data = preprocess(frame, (in_h, in_w), input_dtype)

            # 3) Inference
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()

            # Attempt standard TF Lite detection postprocess outputs ordering
            try:
                boxes = interpreter.get_tensor(output_details[0]["index"])[0]
                classes = interpreter.get_tensor(output_details[1]["index"])[0]
                scores = interpreter.get_tensor(output_details[2]["index"])[0]
                num = interpreter.get_tensor(output_details[3]["index"])[0]
            except Exception:
                # Fallback: re-read in case ordering differs; here we still assume standard 4 outputs.
                outs = [interpreter.get_tensor(od["index"]) for od in output_details]
                # Heuristic assignment
                boxes = None
                classes = None
                scores = None
                num = None
                for out in outs:
                    arr = np.squeeze(out)
                    if arr.ndim == 2 and arr.shape[-1] == 4:
                        boxes = arr
                    elif arr.ndim == 1 and arr.size > 4 and np.issubdtype(arr.dtype, np.floating):
                        # could be scores or classes
                        if scores is None:
                            scores = arr
                        else:
                            classes = arr
                    elif arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
                        num = arr
                if boxes is None or classes is None or scores is None or num is None:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    classes = np.zeros((0,), dtype=np.int32)
                    scores = np.zeros((0,), dtype=np.float32)
                    num = 0

            # Ensure correct types
            classes = classes.astype(np.int32, copy=False)
            detections = postprocess(frame.shape, boxes, classes, scores, num, confidence_threshold)

            # 4) Output handling: update proxy mAP and draw results
            update_ap_data_for_frame(detections, ap_data, num_pos_by_class)
            current_map = compute_map(ap_data, num_pos_by_class)
            draw_detections(frame, detections, labels, map_value=(current_map if current_map is not None else 0.0))

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    final_map = compute_map(ap_data, num_pos_by_class)
    if final_map is None:
        final_map = 0.0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s ({(frame_count/elapsed) if elapsed>0 else 0:.2f} FPS).")
    print(f"Saved output video to: {output_path}")
    print(f"Proxy mAP over the processed video: {final_map:.4f} (note: computed without ground-truth)")


if __name__ == "__main__":
    main()