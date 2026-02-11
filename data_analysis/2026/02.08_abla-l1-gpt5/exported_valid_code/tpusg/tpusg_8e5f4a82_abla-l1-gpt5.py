import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# -----------------------------
# Configuration parameters
# -----------------------------
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

EDGETPU_DELEGATE_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# -----------------------------
# Utilities
# -----------------------------
def load_labels(label_path):
    labels = {}
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if " " in line:
                parts = line.split(maxsplit=1)
                if parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                else:
                    labels[i] = line
            else:
                # Single label per line without explicit id; use line index as id
                labels[i] = line
    return labels

def make_interpreter(model_path, delegate_path):
    return Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(delegate_path)]
    )

def get_output_tensors(interpreter):
    # Typical SSD detection postprocess order: boxes, classes, scores, count
    output_details = interpreter.get_output_details()
    # Attempt to identify tensors by shapes
    idx_boxes, idx_classes, idx_scores, idx_count = None, None, None, None
    for i, od in enumerate(output_details):
        shape = od["shape"]
        dtype = od["dtype"]
        if len(shape) == 3 and shape[-1] == 4:
            idx_boxes = i
        elif len(shape) == 2 and shape[-1] >= 1 and dtype in (np.float32, np.float16):
            # Could be classes or scores; distinguish later by dtype/int
            # Some models output classes as float32; we'll try to pick by name hint if available
            name = od.get("name", "").lower()
            if "score" in name:
                idx_scores = i
            elif "class" in name:
                idx_classes = i
        elif len(shape) == 1 and shape[0] == 1:
            idx_count = i

    # Fallback to common order if identification failed
    if None in (idx_boxes, idx_classes, idx_scores, idx_count):
        if len(output_details) >= 4:
            idx_boxes = 0 if idx_boxes is None else idx_boxes
            idx_classes = 1 if idx_classes is None else idx_classes
            idx_scores = 2 if idx_scores is None else idx_scores
            idx_count = 3 if idx_count is None else idx_count

    boxes = interpreter.get_tensor(output_details[idx_boxes]["index"])
    classes = interpreter.get_tensor(output_details[idx_classes]["index"])
    scores = interpreter.get_tensor(output_details[idx_scores]["index"])
    count = interpreter.get_tensor(output_details[idx_count]["index"])

    # Squeeze to expected shapes
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores).astype(np.float32)
    count = int(np.squeeze(count).astype(np.int32))
    return boxes, classes, scores, count

def preprocess_frame(frame, input_size, input_dtype):
    h, w = input_size
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (w, h))
    if input_dtype == np.float32:
        img_resized = img_resized.astype(np.float32) / 255.0
    else:
        img_resized = img_resized.astype(np.uint8)
    return img_resized

def to_color_for_id(class_id):
    # Deterministic color from class id
    np.random.seed(class_id + 13)
    c = np.random.randint(0, 255, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])

def draw_detections(frame, detections, labels, map_value):
    # detections: list of (ymin, xmin, ymax, xmax, class_id, score)
    for (ymin, xmin, ymax, xmax, cid, score) in detections:
        color = to_color_for_id(cid)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = labels.get(cid, str(cid))
        caption = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(ymin - 8, th + 2)
        cv2.rectangle(frame, (xmin, y_text - th - 4), (xmin + tw + 4, y_text + baseline - 2), color, -1)
        cv2.putText(frame, caption, (xmin + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay mAP (proxy) at top-left
    cv2.putText(frame, f"mAP (approx): {map_value:.3f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)

def compute_proxy_map(scores_by_class):
    # Proxy mAP: mean of per-class average confidence across detected classes
    per_class_avgs = []
    for cid, scores in scores_by_class.items():
        if len(scores) > 0:
            per_class_avgs.append(float(np.mean(scores)))
    if len(per_class_avgs) == 0:
        return 0.0
    return float(np.mean(per_class_avgs))

# -----------------------------
# Main application
# -----------------------------
def main():
    # Load labels
    labels = load_labels(LABEL_PATH)

    # Prepare interpreter with EdgeTPU delegate
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_DELEGATE_PATH)
    interpreter.allocate_tensors()

    # Model input details
    input_details = interpreter.get_input_details()[0]
    input_height, input_width = input_details["shape"][1], input_details["shape"][2]
    input_dtype = input_details["dtype"]

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open input video: {INPUT_PATH}")
        return

    # Prepare output writer
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (in_fps > 0):
        in_fps = 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (in_w, in_h))
    if not writer.isOpened():
        print(f"Error: cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Metrics accumulators
    scores_by_class = {}  # cid -> [scores]
    frame_index = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            # Preprocess
            pre_img = preprocess_frame(frame, (input_height, input_width), input_dtype)
            # Set tensor
            if input_dtype == np.float32:
                input_data = np.expand_dims(pre_img, axis=0).astype(np.float32)
            else:
                input_data = np.expand_dims(pre_img, axis=0).astype(np.uint8)

            interpreter.set_tensor(input_details["index"], input_data)

            # Inference
            interpreter.invoke()

            # Postprocess
            boxes, classes, scores, count = get_output_tensors(interpreter)

            detections = []
            for i in range(count):
                score = float(scores[i])
                if score < CONFIDENCE_THRESHOLD:
                    continue
                cid = int(classes[i])
                ymin, xmin, ymax, xmax = boxes[i]  # normalized
                # Convert to absolute pixel coordinates
                x0 = int(max(0, min(in_w - 1, round(xmin * in_w))))
                y0 = int(max(0, min(in_h - 1, round(ymin * in_h))))
                x1 = int(max(0, min(in_w - 1, round(xmax * in_w))))
                y1 = int(max(0, min(in_h - 1, round(ymax * in_h))))
                # Ensure proper ordering
                xmin_px, xmax_px = sorted([x0, x1])
                ymin_px, ymax_px = sorted([y0, y1])

                detections.append((ymin_px, xmin_px, ymax_px, xmax_px, cid, score))
                if cid not in scores_by_class:
                    scores_by_class[cid] = []
                scores_by_class[cid].append(score)

            # Compute proxy mAP so far
            map_value = compute_proxy_map(scores_by_class)

            # Draw
            draw_detections(frame, detections, labels, map_value)

            # Write frame
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    fps = frame_index / total_time if total_time > 0 else 0.0
    final_map = compute_proxy_map(scores_by_class)

    print("Processing completed.")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_index}")
    print(f"Average FPS: {fps:.2f}")
    print(f"mAP (approx): {final_map:.4f}")

if __name__ == "__main__":
    main()