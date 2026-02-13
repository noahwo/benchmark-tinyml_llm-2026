import os
import time
import numpy as np
import cv2

# Per guideline, use ai_edge_litert Interpreter
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP
CONFIDENCE_THRESHOLD = 0.5

# =========================
# UTILITY FUNCTIONS
# =========================
def load_labels(label_path):
    # Assumes one label per line. If a background label like "???", keep it so indices match.
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line)
    return labels

def preprocess_frame(frame_bgr, input_height, input_width, input_dtype):
    # Convert BGR to RGB, resize to model input, add batch dimension
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height))
    if input_dtype == np.uint8:
        input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
    else:
        # Float32: normalize to [0,1]
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return input_data

def get_detections(interpreter, threshold, frame_w, frame_h, labels):
    # TFLite SSD output order: boxes, classes, scores, num
    out_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(out_details[0]['index'])[0]   # [N,4] in [ymin, xmin, ymax, xmax], normalized
    classes = interpreter.get_tensor(out_details[1]['index'])[0] # [N]
    scores = interpreter.get_tensor(out_details[2]['index'])[0]  # [N]
    num = int(interpreter.get_tensor(out_details[3]['index'])[0])

    results = []
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        # Convert to absolute pixel coordinates and clip
        x1 = max(0, int(xmin * frame_w))
        y1 = max(0, int(ymin * frame_h))
        x2 = min(frame_w - 1, int(xmax * frame_w))
        y2 = min(frame_h - 1, int(ymax * frame_h))
        cls_id = int(classes[i])  # Typically 0-based or 1-based depending on label file
        label = labels[cls_id] if 0 <= cls_id < len(labels) else f"id_{cls_id}"
        results.append((x1, y1, x2, y2, cls_id, label, score))
    return results

def draw_detections(frame_bgr, detections, mAP_value):
    # Draw boxes and labels; overlay mAP
    for (x1, y1, x2, y2, cls_id, label, score) in detections:
        color = ((37 * (cls_id + 1)) % 255, (17 * (cls_id + 1)) % 255, (29 * (cls_id + 1)) % 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - 8)
        cv2.rectangle(frame_bgr, (x1, y_text - th - 4), (x1 + tw + 2, y_text + 2), color, -1)
        cv2.putText(frame_bgr, text, (x1 + 1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw mAP at top-left
    map_text = f"mAP: {mAP_value:.3f}"
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

def safe_video_props(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        # Fallback probe using first frame
        ret, frame = cap.read()
        if not ret:
            return 30.0, 640, 480, None
        height, width = frame.shape[:2]
        return fps, width, height, frame
    return fps, width, height, None

# =========================
# MAIN APPLICATION LOGIC
# =========================
def main():
    # 1) Setup: load interpreter, allocate tensors, load labels, open input video
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

    labels = load_labels(LABEL_PATH)

    # Initialize interpreter (use available CPU cores on Raspberry Pi 4B)
    num_threads = max(1, (os.cpu_count() or 4))
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()
    in_index = in_details[0]['index']
    in_shape = in_details[0]['shape']  # [1, H, W, C]
    in_dtype = in_details[0]['dtype']
    input_h, input_w = int(in_shape[1]), int(in_shape[2])

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps, frame_w, frame_h, first_frame = safe_video_props(cap)

    # 2-pass approach:
    #   Pass 1: Run inference per frame, collect detections, and accumulate scores to compute mAP proxy.
    #   Pass 2: Reopen video, draw detections and final mAP, write to output.

    detections_per_frame = []
    all_scores = []
    frame_idx = 0

    # If we probed and consumed first frame, handle it
    frames_to_process = []
    if first_frame is not None:
        frames_to_process.append(first_frame)

    # Read remaining frames in pass 1
    while True:
        if first_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            frames_to_process.append(frame)
        # Process accumulated frames (to avoid storing all video frames in memory, we process immediately)
        for frame in frames_to_process:
            # 2) Preprocessing
            input_data = preprocess_frame(frame, input_h, input_w, in_dtype)
            interpreter.set_tensor(in_index, input_data)

            # 3) Inference
            t0 = time.time()
            interpreter.invoke()
            _ = time.time() - t0  # inference_time (not strictly required to display)

            # 4) Collect detections for this frame
            dets = get_detections(interpreter, CONFIDENCE_THRESHOLD, frame_w, frame_h, labels)
            detections_per_frame.append(dets)
            # Accumulate scores for proxy mAP computation
            for d in dets:
                all_scores.append(d[6])  # score at index 6

            frame_idx += 1
        frames_to_process = []
        first_frame = None

    cap.release()

    # Compute proxy mAP: mean of detection confidences across all detections above threshold
    # Note: True mAP requires ground-truth annotations to compute precision/recall vs IoU.
    # Here we provide a proxy metric due to unavailable ground truth.
    if len(all_scores) > 0:
        mAP_value = float(np.mean(np.array(all_scores, dtype=np.float32)))
    else:
        mAP_value = 0.0

    # Pass 2: Draw and save the video with rectangles, labels, and mAP text
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to reopen input video for rendering: {INPUT_PATH}")

    out_index = 0
    while True:
        ret, frame = cap.read()
        if not ret or out_index >= len(detections_per_frame):
            break
        dets = detections_per_frame[out_index]
        draw_detections(frame, dets, mAP_value)
        writer.write(frame)
        out_index += 1

    cap.release()
    writer.release()

    # Optional console summary
    print(f"Processed {out_index} frames.")
    print(f"Detections above threshold: {len(all_scores)}")
    print(f"mAP (proxy mean confidence): {mAP_value:.4f}")
    print(f"Saved output video to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()