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
# Helper Functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    # Try to parse "index label" or "index: label"; fallback to enumerated lines
    has_index = True
    for i, line in enumerate(lines):
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip()
            if left.isdigit():
                labels[int(left)] = right
            else:
                has_index = False
                break
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                has_index = False
                break
    if not has_index:
        # Fallback: one label per line, enumerate starting at 0
        labels = {i: line for i, line in enumerate(lines) if line}
    return labels

def make_interpreter(model_path):
    delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    delegates = []
    if os.path.exists(delegate_path):
        delegates.append(load_delegate(delegate_path))
    else:
        # Fallback to default name if absolute path not present
        delegates.append(load_delegate("libedgetpu.so.1.0"))
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame_bgr, input_details):
    # Convert BGR to RGB and resize to model input shape
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (w, h))
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        input_data = np.asarray(resized, dtype=np.float32) / 255.0
    else:
        # uint8 input
        input_data = np.asarray(resized, dtype=np.uint8)
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def get_output_tensors(interpreter):
    # Try common ordering for SSD models compiled for EdgeTPU:
    # [boxes, classes, scores, num]
    output_details = interpreter.get_output_details()
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = interpreter.get_tensor(output_details[3]['index'])
        return boxes, classes, scores, num
    except Exception:
        # Fallback: identify by shapes and value ranges
        outs = [interpreter.get_tensor(od['index']) for od in output_details]
        boxes = None
        classes = None
        scores = None
        num = None
        for arr in outs:
            arr_squeezed = np.squeeze(arr)
            if arr_squeezed.ndim == 2 and arr_squeezed.shape[-1] == 4:
                boxes = arr
        for arr in outs:
            arr_squeezed = np.squeeze(arr)
            if arr_squeezed.ndim == 1 and arr_squeezed.size == 1:
                num = arr
        # scores: values in [0,1]
        for arr in outs:
            vals = np.squeeze(arr).astype(np.float32)
            if vals.ndim == 1 and vals.size > 1:
                # Not boxes or num; could be scores or classes
                if np.all((vals >= 0.0) & (vals <= 1.0)):
                    scores = arr
        # classes: remaining
        for arr in outs:
            if arr is boxes or arr is scores or arr is num:
                continue
            arr_squeezed = np.squeeze(arr)
            if arr_squeezed.ndim == 1 and arr_squeezed.size > 1:
                classes = arr
        if boxes is None or classes is None or scores is None or num is None:
            raise RuntimeError("Unable to parse model outputs.")
        return boxes, classes, scores, num

def postprocess_detections(boxes, classes, scores, num, orig_w, orig_h, labels, threshold):
    detections = []
    # Squeeze to remove batch dimension if present
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    if np.ndim(num) > 0:
        num = int(np.squeeze(num).astype(np.int32))
    else:
        num = int(num)
    num = min(num, boxes.shape[0], scores.shape[0], classes.shape[0])
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        cls_id = int(classes[i])
        ymin, xmin, ymax, xmax = boxes[i]
        # Boxes are normalized [0,1] relative to model input, map to original frame size
        x1 = int(max(0, min(orig_w - 1, xmin * orig_w)))
        y1 = int(max(0, min(orig_h - 1, ymin * orig_h)))
        x2 = int(max(0, min(orig_w - 1, xmax * orig_w)))
        y2 = int(max(0, min(orig_h - 1, ymax * orig_h)))
        label = labels.get(cls_id, str(cls_id))
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "score": score,
            "class_id": cls_id,
            "label": label
        })
    return detections

def draw_detections(frame, detections, map_text):
    # Simple color palette
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        label = det["label"]
        color = colors[det["class_id"] % len(colors)]
        thickness = max(2, int(round(0.002 * (frame.shape[0] + frame.shape[1]) / 2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        caption = f"{label}: {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # Put mAP text on the frame
    cv2.putText(frame, f"mAP: {map_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)
    return frame

def compute_map_placeholder():
    # Real mAP calculation requires ground-truth annotations to match predictions.
    # Since no ground-truth is provided, we return "N/A".
    return None

# =========================
# Main Pipeline
# =========================
def main():
    # Print descriptions
    print("Application: TFLite object detection with TPU")
    print("Input: Read a single video file from the given input_path")
    print("Output: Output the video file with rectangles on detected objects and mAP text")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    labels = load_labels(label_path)
    print(f"Loaded {len(labels)} labels from {label_path}")

    # Initialize TFLite Interpreter with EdgeTPU
    print("Initializing TFLite Interpreter with EdgeTPU delegate...")
    t0 = time.time()
    interpreter = make_interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Interpreter initialized in {time.time() - t0:.2f}s")
    print(f"Model input shape: {input_details[0]['shape']} dtype: {input_details[0]['dtype']}")

    # Video I/O setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or np.isnan(in_fps) or in_fps <= 0:
        in_fps = 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, in_fps, (in_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # Placeholder mAP (N/A without ground truth)
    overall_map = compute_map_placeholder()
    map_text = "N/A (no GT)" if overall_map is None else f"{overall_map:.3f}"

    frame_index = 0
    total_infer_time = 0.0

    print("Starting inference on video...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Preprocess
        input_data = preprocess_frame(frame_bgr, input_details)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Inference
        t_start = time.time()
        interpreter.invoke()
        infer_time = time.time() - t_start
        total_infer_time += infer_time

        # Postprocess
        boxes, classes, scores, num = get_output_tensors(interpreter)
        detections = postprocess_detections(
            boxes, classes, scores, num,
            orig_w=frame_bgr.shape[1],
            orig_h=frame_bgr.shape[0],
            labels=labels,
            threshold=confidence_threshold
        )

        # Draw and write
        frame_annotated = draw_detections(frame_bgr, detections, map_text)
        writer.write(frame_annotated)

        frame_index += 1
        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames... avg inference {1000.0 * total_infer_time / frame_index:.2f} ms/frame")

    cap.release()
    writer.release()

    if frame_index > 0:
        avg_ms = (total_infer_time / frame_index) * 1000.0
    else:
        avg_ms = 0.0

    print(f"Finished. Frames: {frame_index}, Avg inference time: {avg_ms:.2f} ms/frame")
    print(f"Saved output video to: {output_path}")

if __name__ == "__main__":
    main()