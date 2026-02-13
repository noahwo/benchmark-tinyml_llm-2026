import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

INPUT_DESCRIPTION = "Read a single video file from the given input_path"
OUTPUT_DESCRIPTION = "Output the video file with rectangles drew on the detected objects, along with texts of labels and calculated mAP(mean average precision)"

EDGETPU_DELEGATE_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


# =========================
# Utilities
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print("Label file not found:", path)
        return labels
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try "id: label"
            if ":" in line:
                left, right = line.split(":", 1)
                left = left.strip()
                right = right.strip()
                try:
                    idx = int(left)
                    labels[idx] = right
                    continue
                except ValueError:
                    pass
            # Try "id label"
            parts = line.split()
            if parts:
                try:
                    idx = int(parts[0])
                    name = " ".join(parts[1:]).strip()
                    labels[idx] = name if name else str(idx)
                    continue
                except ValueError:
                    pass
            # Fallback: sequential
            labels[i] = line
    return labels


def make_interpreter(model_path, delegate_path):
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(delegate_path)]
        )
        return interpreter
    except Exception as e:
        print("Failed to load EdgeTPU delegate:", e)
        raise


def preprocess_frame(frame_bgr, input_details):
    # Converts BGR frame to model input tensor shape and dtype
    h, w = input_details["shape"][1], input_details["shape"][2]
    dtype = input_details["dtype"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    if dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    else:
        # Assume uint8 quantized input (common for EdgeTPU models)
        inp = resized.astype(np.uint8)

    inp = np.expand_dims(inp, axis=0)
    return inp


def parse_detections(interpreter, frame_w, frame_h, score_threshold):
    # Extracts outputs and returns a list of detections
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(d["index"]) for d in output_details]

    boxes = None
    classes = None
    scores = None
    num = None

    # Heuristics for common TFLite detection head layout
    for arr in outputs:
        a = np.squeeze(arr)
        if a.ndim == 2 and a.shape[-1] == 4:
            boxes = a  # [N,4] in [ymin, xmin, ymax, xmax] normalized
        elif a.ndim == 1 and a.size > 4:
            # Could be scores or classes
            maxv = float(np.max(a)) if a.size else 0.0
            minv = float(np.min(a)) if a.size else 0.0
            if 0.0 <= minv and maxv <= 1.0:
                scores = a.astype(np.float32)
            else:
                classes = a.astype(np.int32)
        elif a.ndim == 0:
            # num_detections scalar
            try:
                num = int(a)
            except Exception:
                pass

    if boxes is None:
        # Some models keep batch dim on boxes: [1,N,4]
        for arr in outputs:
            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] == 4:
                boxes = arr[0]
                break

    if scores is None:
        # Try to find scores in the 2D [1,N] layout
        for arr in outputs:
            if arr.ndim == 2 and arr.shape[0] == 1:
                cand = arr[0]
                maxv = float(np.max(cand)) if cand.size else 0.0
                minv = float(np.min(cand)) if cand.size else 0.0
                if 0.0 <= minv and maxv <= 1.0:
                    scores = cand.astype(np.float32)
                    break

    if classes is None:
        for arr in outputs:
            if arr.ndim == 2 and arr.shape[0] == 1:
                cand = arr[0]
                maxv = float(np.max(cand)) if cand.size else 0.0
                minv = float(np.min(cand)) if cand.size else 0.0
                if not (0.0 <= minv and maxv <= 1.0):
                    classes = cand.astype(np.int32)
                    break

    if num is None:
        # Fallback to lengths
        if scores is not None:
            num = int(scores.shape[0])
        elif boxes is not None:
            num = int(boxes.shape[0])
        else:
            num = 0

    detections = []
    if boxes is None or scores is None:
        return detections

    for i in range(num):
        score = float(scores[i])
        if score < score_threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        # Convert to pixel coordinates
        x1 = max(0, min(int(x_min * frame_w), frame_w - 1))
        y1 = max(0, min(int(y_min * frame_h), frame_h - 1))
        x2 = max(0, min(int(x_max * frame_w), frame_w - 1))
        y2 = max(0, min(int(y_max * frame_h), frame_h - 1))
        cls_id = int(classes[i]) if classes is not None and i < len(classes) else -1

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": cls_id
        })
    return detections


def draw_detections(frame_bgr, detections, labels, map_value):
    # Draw bounding boxes and labels on frame
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cid = det["class_id"]
        label = labels.get(cid, str(cid)) if cid is not None else "N/A"
        color = (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame_bgr, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw running mAP (proxy due to missing ground truth)
    map_text = f"mAP (proxy): {map_value:.3f}"
    (mw, mh), mbl = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (10, 10), (10 + mw + 10, 10 + mh + 10), (0, 0, 0), -1)
    cv2.putText(frame_bgr, map_text, (15, 10 + mh), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr


def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# =========================
# Main Application
# =========================
def main():
    print("Application: TFLite object detection with TPU")
    print("Target device: Google Coral Dev Board")
    print("Input:", INPUT_DESCRIPTION)
    print("Output:", OUTPUT_DESCRIPTION)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Create interpreter with EdgeTPU delegate
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_DELEGATE_PATH)
    interpreter.allocate_tensors()

    # Get model input details
    input_details = interpreter.get_input_details()[0]
    # output_details = interpreter.get_output_details()  # Not used directly; parsed via helper

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Failed to open input video:", INPUT_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # default fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Failed to open output for writing:", OUTPUT_PATH)
        cap.release()
        return

    # For proxy mAP computation (no ground-truth available)
    # We approximate mAP as mean confidence of all detections above threshold across the video.
    conf_sum = 0.0
    conf_count = 0

    frame_index = 0
    t0 = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Preprocess
            inp = preprocess_frame(frame_bgr, input_details)

            # Feed input
            interpreter.set_tensor(input_details["index"], inp)

            # Inference
            interpreter.invoke()

            # Parse detections
            detections = parse_detections(interpreter, width, height, CONFIDENCE_THRESHOLD)

            # Update proxy mAP
            for det in detections:
                conf_sum += det["score"]
                conf_count += 1
            map_proxy = (conf_sum / conf_count) if conf_count > 0 else 0.0

            # Draw and write frame
            annotated = draw_detections(frame_bgr, detections, labels, map_proxy)
            writer.write(annotated)

            frame_index += 1

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    final_map = (conf_sum / conf_count) if conf_count > 0 else 0.0
    print(f"Processed {frame_index} frames in {elapsed:.2f}s ({(frame_index/elapsed if elapsed > 0 else 0):.2f} FPS).")
    print(f"Saved annotated output to: {OUTPUT_PATH}")
    print(f"Proxy mAP over video (mean confidence above {CONFIDENCE_THRESHOLD}): {final_map:.4f}")


if __name__ == "__main__":
    main()