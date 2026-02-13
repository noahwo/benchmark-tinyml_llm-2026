import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"  # corrected path
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# Helper functions
# =========================
def load_labels(path):
    """
    Load labels from a label map file.
    Supports the following formats per line:
      - "id label"
      - "id: label"
      - "label" (implicit incremental id starting at 0)
    Returns dict: {id(int): label(str)}
    """
    labels = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Try "id: label"
                if ':' in line:
                    left, right = line.split(':', 1)
                    left = left.strip()
                    right = right.strip()
                    if left.isdigit():
                        labels[int(left)] = right
                        continue
                # Try "id label"
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                    continue
                # Fallback: implicit id
                labels[idx] = line
    except Exception as e:
        print(f"Warning: Failed to load labels from {path}: {e}")
    return labels

def make_interpreter(model_file, delegate_lib="/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"):
    """
    Create a TFLite interpreter with EdgeTPU delegate.
    """
    try:
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=[load_delegate(delegate_lib)]
        )
    except ValueError as e:
        raise RuntimeError(f"Failed to load EdgeTPU delegate '{delegate_lib}': {e}")
    return interpreter

def preprocess_frame(frame_bgr, input_size, input_dtype):
    """
    Resize and convert BGR frame to model's expected input tensor.
    - Convert BGR to RGB.
    - Resize to input_size (width, height).
    - If input_dtype is float32, normalize to [0,1].
    - Return a numpy array of shape (1, height, width, 3) with proper dtype.
    """
    ih, iw = input_size[1], input_size[0]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)

    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def get_output(interpreter):
    """
    Extract common object detection model outputs:
      - boxes: [N, 4] in normalized coordinates [ymin, xmin, ymax, xmax]
      - classes: [N] float class indices
      - scores: [N] float confidence scores
      - count: number of detections (int)
    """
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    count = interpreter.get_tensor(output_details[3]['index'])

    # Squeeze batch dimension
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)
    if np.isscalar(count):
        num = int(count)
    else:
        num = int(np.squeeze(count))
    return boxes, classes, scores, num

def draw_detections(frame, detections, labels, mAP_value):
    """
    Draw bounding boxes and labels on frame.
    detections: list of dicts with keys: 'box' (xmin, ymin, xmax, ymax), 'score', 'class_id'
    labels: dict {id: label}
    mAP_value: float, will be displayed on frame as "mAP"
    """
    h, w = frame.shape[:2]

    for det in detections:
        xmin, ymin, xmax, ymax = det['box']
        score = det['score']
        class_id = det['class_id']
        label = labels.get(class_id, f"class_{class_id}")
        caption = f"{label}: {score:.2f}"

        # Choose a color based on class_id for consistency
        color = (
            int((37 * (class_id + 1)) % 255),
            int((17 * (class_id + 1)) % 255),
            int((29 * (class_id + 1)) % 255),
        )

        # Draw rectangle
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Draw filled rectangle for text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, max(0, ymin - th - baseline - 4)),
                             (xmin + tw + 4, ymin), color, thickness=-1)
        # Put text
        cv2.putText(frame, caption, (xmin + 2, ymin - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw mAP on the frame (top-left corner)
    cv2.putText(frame, f"mAP (proxy): {mAP_value*100:.2f}%",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 200, 10), 2, cv2.LINE_AA)

def clip_bbox(xmin, ymin, xmax, ymax, w, h):
    """
    Clip bounding box coordinates to image dimensions.
    """
    xmin = max(0, min(w - 1, xmin))
    ymin = max(0, min(h - 1, ymin))
    xmax = max(0, min(w - 1, xmax))
    ymax = max(0, min(h - 1, ymax))
    return xmin, ymin, xmax, ymax

def compute_proxy_map(class_to_scores):
    """
    Compute a proxy mAP metric without ground-truth:
    - For each class with any detections, AP_c = mean(confidence scores of detections above threshold).
    - mAP = mean(AP_c) across classes with any detections.
    NOTE: This is NOT a true mAP since no ground-truth is provided.
    """
    ap_values = []
    for _, scores in class_to_scores.items():
        if len(scores) > 0:
            ap_values.append(float(np.mean(np.array(scores, dtype=np.float32))))
    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(np.array(ap_values, dtype=np.float32)))

# =========================
# Main application
# =========================
def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(label_path)
    if not labels:
        print("Warning: No labels were loaded; classes will be shown as numeric IDs.")

    # Load interpreter with EdgeTPU delegate
    print("Loading TFLite model with EdgeTPU delegate...")
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    _, in_h, in_w, in_c = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    print(f"Model input: shape=({in_h}, {in_w}, {in_c}), dtype={input_dtype}")
    print(f"Model outputs: {len(output_details)} tensors")

    # Video I/O setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    print(f"Processing video: {input_path}")
    print(f"Saving annotated video to: {output_path}")
    print(f"Frame size: {width}x{height}, FPS: {fps}")

    # Metrics containers
    class_to_scores = {}  # class_id -> list of confidences
    total_frames = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            # Preprocess
            input_tensor = preprocess_frame(frame, (in_w, in_h), input_dtype)

            # Inference
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()

            # Post-process
            boxes, classes, scores, count = get_output(interpreter)
            detections = []
            for i in range(count):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                class_id = int(classes[i])

                # boxes are [ymin, xmin, ymax, xmax] in normalized coordinates
                ymin = boxes[i][0]
                xmin = boxes[i][1]
                ymax = boxes[i][2]
                xmax = boxes[i][3]

                # Convert to pixel coordinates
                x0 = int(xmin * width)
                y0 = int(ymin * height)
                x1 = int(xmax * width)
                y1 = int(ymax * height)

                x0, y0, x1, y1 = clip_bbox(x0, y0, x1, y1, width, height)
                if x1 <= x0 or y1 <= y0:
                    continue

                detections.append({
                    'box': (x0, y0, x1, y1),
                    'score': score,
                    'class_id': class_id
                })

                # Update proxy AP metrics store
                if class_id not in class_to_scores:
                    class_to_scores[class_id] = []
                class_to_scores[class_id].append(score)

            # Compute running proxy mAP
            proxy_map = compute_proxy_map(class_to_scores)

            # Draw and write frame
            draw_detections(frame, detections, labels, proxy_map)
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    proxy_map_final = compute_proxy_map(class_to_scores)
    print(f"Done. Processed {total_frames} frames in {elapsed:.2f}s "
          f"({(total_frames / elapsed) if elapsed > 0 else 0:.2f} FPS).")
    print(f"Final proxy mAP (no ground-truth): {proxy_map_final*100:.2f}%")

if __name__ == "__main__":
    """
    TFLite object detection with TPU
    - Setup: load TFLite interpreter with EdgeTPU and labels, open input video.
    - Preprocessing: resize and normalize frames as required by the model.
    - Inference: run model per frame, retrieve boxes/classes/scores.
    - Output: draw detection boxes with labels and confidence, compute a proxy mAP
      (note: without ground-truth annotations, this is an approximation based on mean
       detection confidence per class), and save annotated video.
    """
    main()