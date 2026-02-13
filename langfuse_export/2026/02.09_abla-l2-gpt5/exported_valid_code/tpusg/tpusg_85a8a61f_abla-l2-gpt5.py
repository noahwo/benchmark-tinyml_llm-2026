import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

def load_labels(label_path):
    labels = {}
    if not os.path.isfile(label_path):
        print(f"Warning: Label file not found at {label_path}. Proceeding without labels.")
        return labels
    with open(label_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Try common formats: "0 label", "0: label", "label"
            idx = None
            name = None
            parts = line.split()
            # Try "0 label" format
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0])
                name = " ".join(parts[1:])
            else:
                # Try "0: label" format
                if ":" in line:
                    left, right = line.split(":", 1)
                    left = left.strip()
                    right = right.strip()
                    if left.isdigit():
                        idx = int(left)
                        name = right
                # Fallback: plain list, use line number as index
                if idx is None:
                    idx = i
                    name = line
            labels[idx] = name
    return labels

def make_interpreter(model_path, edgetpu_lib):
    delegate = load_delegate(edgetpu_lib)
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    height, width = input_details['shape'][1], input_details['shape'][2]
    dtype = input_details['dtype']
    return input_details['index'], height, width, dtype

def get_output_indices(interpreter):
    # Detect the common detection postprocess outputs
    # We will map them by shape:
    # boxes: [1, N, 4]
    # classes: [1, N]
    # scores: [1, N]
    # count: [1]
    outs = interpreter.get_output_details()
    idx_boxes = idx_classes = idx_scores = idx_count = None
    for od in outs:
        shp = od['shape']
        if len(shp) == 3 and shp[-1] == 4:
            idx_boxes = od['index']
        elif len(shp) == 2:
            # Could be classes or scores
            if 'quantization_parameters' in od and od['dtype'] == np.float32:
                # Can't differentiate reliably; use name hints if present
                name = od.get('name', '').lower()
                if 'score' in name:
                    idx_scores = od['index']
                elif 'class' in name:
                    idx_classes = od['index']
                else:
                    # Assign later if still None
                    if idx_scores is None:
                        idx_scores = od['index']
                    else:
                        idx_classes = od['index']
            else:
                # Fallback
                name = od.get('name', '').lower()
                if 'score' in name:
                    idx_scores = od['index']
                elif 'class' in name:
                    idx_classes = od['index']
                else:
                    if idx_scores is None:
                        idx_scores = od['index']
                    else:
                        idx_classes = od['index']
        elif len(shp) == 1 and shp[0] == 1:
            idx_count = od['index']
    return idx_boxes, idx_classes, idx_scores, idx_count

def preprocess_frame(frame, input_h, input_w, input_dtype):
    # Resize to model input size
    resized = cv2.resize(frame, (input_w, input_h))
    if input_dtype == np.float32:
        # Normalize to [0,1]
        input_data = resized.astype(np.float32) / 255.0
    else:
        # Assume uint8
        input_data = resized.astype(np.uint8)
    # Add batch dimension
    return np.expand_dims(input_data, axis=0)

def draw_detections(frame, detections, labels, color=(0, 255, 0)):
    # detections: list of dict with keys 'bbox', 'score', 'class_id'
    h, w = frame.shape[:2]
    for det in detections:
        y_min, x_min, y_max, x_max = det['bbox']
        # Coordinates are normalized [0,1]; map to frame size
        x1 = max(0, min(w - 1, int(x_min * w)))
        y1 = max(0, min(h - 1, int(y_min * h)))
        x2 = max(0, min(w - 1, int(x_max * w)))
        y2 = max(0, min(h - 1, int(y_max * h)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        class_id = int(det['class_id'])
        score = det['score']
        label_text = labels.get(class_id, str(class_id))
        caption = f"{label_text}: {score:.2f}"
        # Put label above box if possible
        y_text = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, caption, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    # Validate input file
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU delegate
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_SHARED_LIB)
    input_index, in_h, in_w, in_dtype = get_input_details(interpreter)
    out_idx_boxes, out_idx_classes, out_idx_scores, out_idx_count = get_output_indices(interpreter)

    if None in (out_idx_boxes, out_idx_classes, out_idx_scores, out_idx_count):
        raise RuntimeError("Failed to identify all required output tensors from the model.")

    # Setup video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or np.isnan(in_fps) or in_fps <= 0:
        in_fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, in_fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # Metrics accumulators
    total_scores_sum = 0.0
    total_detections_count = 0
    frame_count = 0
    fps_smooth = None
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        input_data = preprocess_frame(frame, in_h, in_w, in_dtype)
        # Set input tensor
        interpreter.set_tensor(input_index, input_data)

        # Inference
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        dt = t1 - t0
        # Smooth FPS estimate
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        if fps_smooth is None:
            fps_smooth = inst_fps
        else:
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

        # Extract outputs
        boxes = interpreter.get_tensor(out_idx_boxes)[0]  # [N,4] in y_min, x_min, y_max, x_max
        classes = interpreter.get_tensor(out_idx_classes)[0]  # [N]
        scores = interpreter.get_tensor(out_idx_scores)[0]  # [N]
        count = int(interpreter.get_tensor(out_idx_count)[0])

        # Build detections list with thresholding
        detections = []
        for i in range(count):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls = int(classes[i])
            bbox = boxes[i]  # normalized
            # Ensure bbox in [0,1]
            y_min, x_min, y_max, x_max = bbox
            y_min = max(0.0, min(1.0, y_min))
            x_min = max(0.0, min(1.0, x_min))
            y_max = max(0.0, min(1.0, y_max))
            x_max = max(0.0, min(1.0, x_max))
            detections.append({'bbox': (y_min, x_min, y_max, x_max), 'score': score, 'class_id': cls})

        # Update proxy mAP metric (mean of scores over all detections so far)
        if detections:
            scores_this_frame = [d['score'] for d in detections]
            total_scores_sum += float(np.sum(scores_this_frame))
            total_detections_count += len(scores_this_frame)
        proxy_map = (total_scores_sum / total_detections_count) if total_detections_count > 0 else 0.0

        # Draw detections and overlay metrics
        draw_detections(frame, detections, labels, color=(0, 255, 0))
        # Overlays: FPS and mAP
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"mAP: {proxy_map:.3f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        # Write to output
        writer.write(frame)

        # Progress logging every 50 frames
        if frame_count % 50 == 0:
            now = time.time()
            elapsed = now - t_prev
            t_prev = now
            print(f"Processed {frame_count} frames | approx {50.0/elapsed:.2f} FPS | current detections: {len(detections)}")

    # Cleanup
    cap.release()
    writer.release()

    print("Processing complete.")
    print(f"Total frames: {frame_count}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Final mAP (proxy): { (total_scores_sum / total_detections_count) if total_detections_count > 0 else 0.0 :.4f}")

if __name__ == "__main__":
    main()