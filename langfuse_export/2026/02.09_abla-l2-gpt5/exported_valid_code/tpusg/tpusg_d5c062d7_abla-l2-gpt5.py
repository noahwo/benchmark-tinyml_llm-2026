import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

def load_labels(label_path):
    labels = {}
    with open(label_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Support "id label" or just "label" formats
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
            else:
                labels[idx] = line
    return labels

def get_interpreter(model_path, edgetpu_lib):
    # Initialize TFLite interpreter with EdgeTPU delegate
    return Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(edgetpu_lib)]
    )

def preprocess_frame(frame, input_size, input_dtype):
    h_in, w_in = input_size
    # BGR to RGB, resize to model's input size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (w_in, h_in))
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(input_dtype)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def parse_outputs(interpreter, output_details):
    boxes = None
    classes = None
    scores = None
    count = None
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        # Typical TF Lite detection output tensors:
        # boxes: [1, num, 4]; classes: [1, num]; scores: [1, num]; count: [1]
        if data.ndim == 3 and data.shape[-1] == 4:
            boxes = data[0]
        elif data.ndim == 2 and data.shape[0] == 1:
            vec = data[0]
            # Heuristic to distinguish classes vs scores
            if np.max(vec) <= 1.0:
                scores = vec
            else:
                classes = vec
        elif data.size == 1:
            try:
                count = int(np.squeeze(data))
            except Exception:
                count = None
    # Fallbacks if count is None
    if scores is not None and count is None:
        count = scores.shape[0]
    return boxes, classes, scores, count

def class_color(class_id):
    # Deterministic pseudo-random color per class id
    rng = np.random.default_rng(class_id)
    color = rng.integers(0, 256, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])

def draw_label_with_background(image, text, org, color, font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # Draw filled rectangle for text background
    cv2.rectangle(image, (x, y - th - baseline), (x + tw, y + baseline), (0, 0, 0), -1)
    # Put text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def main():
    # CONFIGURATION PARAMETERS
    model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
    label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
    input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
    output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
    confidence_threshold = 0.5
    edgetpu_lib = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

    # Sanity checks
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label map not found at: {label_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found at: {input_path}")

    # Setup: load labels
    labels = load_labels(label_path)

    # Setup: load interpreter with EdgeTPU delegate
    interpreter = get_interpreter(model_path, edgetpu_lib)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_h = int(input_details[0]['shape'][1])
    input_w = int(input_details[0]['shape'][2])
    input_dtype = input_details[0]['dtype']

    # Read input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Prepare VideoWriter for output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # Processing loop
    frame_idx = 0
    all_drawn_scores = []  # We'll compute a proxy mAP as the running mean confidence of drawn detections

    start_time_overall = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Preprocess
        input_data = preprocess_frame(frame, (input_h, input_w), input_dtype)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        inf_start = time.time()
        interpreter.invoke()
        inf_end = time.time()

        # Parse outputs
        boxes, classes, scores, count = parse_outputs(interpreter, output_details)
        if boxes is None or classes is None or scores is None or count is None:
            # If output format not recognized, skip drawing
            writer.write(frame)
            continue

        # Draw detections above threshold
        drawn = 0
        for i in range(count):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            # Convert normalized [0,1] coords to pixel coords in original frame
            left = max(0, min(orig_w - 1, int(xmin * orig_w)))
            top = max(0, min(orig_h - 1, int(ymin * orig_h)))
            right = max(0, min(orig_w - 1, int(xmax * orig_w)))
            bottom = max(0, min(orig_h - 1, int(ymax * orig_h)))

            # Draw bounding box
            cid = int(classes[i])
            color = class_color(cid)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Prepare label text
            label_text = labels.get(cid, str(cid))
            text = f"{label_text}: {score:.2f}"
            # Draw label background with text (top-left of box)
            text_x = left
            text_y = max(10, top - 5)
            draw_label_with_background(frame, text, (text_x, text_y), (255, 255, 255), font_scale=0.5, thickness=1)

            all_drawn_scores.append(score)
            drawn += 1

        # Compute a running "mAP" proxy as mean confidence of drawn detections
        map_proxy = float(np.mean(all_drawn_scores)) if all_drawn_scores else 0.0

        # Overlay inference time and mAP on frame
        info_color = (255, 255, 255)
        info_bg = (0, 0, 0)
        header_text = f"Detections: {drawn} | Inference: {(inf_end - inf_start)*1000:.1f} ms | mAP: {map_proxy*100:.2f}%"
        # Draw header background
        (tw, th), baseline = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (5, 5), (5 + tw + 10, 5 + th + baseline + 10), info_bg, -1)
        cv2.putText(frame, header_text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2, cv2.LINE_AA)

        # Write frame to output
        writer.write(frame)

    # Cleanup
    cap.release()
    writer.release()
    elapsed = time.time() - start_time_overall

    # Final report (printed to stdout)
    total_frames = max(0, frame_idx)
    print("Processing complete.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"Total time: {elapsed:.2f} s")
    if all_drawn_scores:
        print(f"mAP (proxy, mean confidence of drawn detections): {np.mean(all_drawn_scores)*100:.2f}%")
    else:
        print("mAP (proxy): N/A (no detections above threshold)")

if __name__ == "__main__":
    main()