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
    """
    Loads labels from file.
    Supports:
    - 'id label' or 'id: label'
    - one label per line (index = line number)
    """
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try "id: label" or "id label"
            if ":" in line:
                parts = line.split(":", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                labels[int(parts[0].strip())] = parts[1].strip()
            else:
                # Fallback: one label per line
                labels[idx] = line
    return labels

def make_interpreter(model_path):
    """
    Initialize TFLite interpreter with EdgeTPU delegate.
    """
    delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    delegates = []
    if os.path.exists(delegate_path):
        try:
            delegates = [load_delegate(delegate_path)]
        except Exception as e:
            print("WARNING: Failed to load EdgeTPU delegate, falling back to CPU. Error:", e)
    else:
        print("WARNING: EdgeTPU delegate library not found, falling back to CPU.")
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    return interpreter

def get_output_tensors(interpreter):
    """
    Retrieve output tensor dict for convenience.
    Expected outputs for TFLite detection postprocess:
    - boxes: [1, num, 4] with values in [0,1] as [ymin, xmin, ymax, xmax]
    - classes: [1, num]
    - scores: [1, num]
    - count: [1]
    """
    output_details = interpreter.get_output_details()
    boxes, classes, scores, count = None, None, None, None
    for od in output_details:
        shape = od.get('shape', [])
        name = od.get('name', '').lower()
        if len(shape) == 3 and shape[-1] == 4:
            boxes = interpreter.get_tensor(od['index'])
        elif len(shape) == 2:
            if 'class' in name:
                classes = interpreter.get_tensor(od['index'])
            elif 'score' in name or 'confidence' in name:
                scores = interpreter.get_tensor(od['index'])
        elif len(shape) == 1 and shape[0] == 1:
            count = interpreter.get_tensor(od['index'])
    # Some models may require fetching in a specific order; if not fetched yet, fetch now
    if boxes is None or classes is None or scores is None or count is None:
        # Try by positional assumption
        tensors = [interpreter.get_tensor(od['index']) for od in output_details]
        # Heuristics
        for t in tensors:
            if t.ndim == 3 and t.shape[-1] == 4:
                boxes = t
        for t in tensors:
            if t.ndim == 2 and boxes is not None and t.shape[1] == boxes.shape[1]:
                # Determine if classes or scores by dtype
                if t.dtype == np.float32 and (scores is None):
                    scores = t
                else:
                    classes = t
        for t in tensors:
            if t.ndim == 1 and t.shape[0] == 1:
                count = t
    return boxes, classes, scores, count

def preprocess_frame(frame_bgr, input_size, input_dtype, input_quant=None):
    """
    Resize, color convert, and format frame for TFLite input.
    Returns input tensor ready to set.
    """
    h_in, w_in = input_size
    # Convert BGR to RGB and resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_in, h_in))
    if input_dtype == np.uint8:
        input_tensor = np.expand_dims(resized.astype(np.uint8), axis=0)
        if input_quant:
            # For quantized models, usually no further scaling needed, but ensure uint8
            pass
    else:
        # Assume float input, normalize to [0,1]
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return input_tensor

def scale_box_to_frame(box, frame_w, frame_h):
    """
    Scale normalized box [ymin, xmin, ymax, xmax] to frame coordinates.
    """
    ymin, xmin, ymax, xmax = box
    x1 = max(0, min(frame_w - 1, int(xmin * frame_w)))
    y1 = max(0, min(frame_h - 1, int(ymin * frame_h)))
    x2 = max(0, min(frame_w - 1, int(xmax * frame_w)))
    y2 = max(0, min(frame_h - 1, int(ymax * frame_h)))
    return x1, y1, x2, y2

def draw_detections(frame, detections, labels):
    """
    Draw bounding boxes and labels on frame.
    detections: list of dicts with keys: class_id, score, box (x1,y1,x2,y2)
    """
    for det in detections:
        x1, y1, x2, y2 = det['box']
        class_id = det['class_id']
        score = det['score']
        label = labels.get(class_id, str(class_id))
        caption = f"{label}: {score:.2f}"
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        # Label background
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 180, 255), -1)
        cv2.putText(frame, caption, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def compute_proxy_map(running_scores_per_class):
    """
    Proxy mAP computation in absence of ground-truth:
    - For each class, AP is approximated as the mean confidence across all detections seen so far.
    - mAP is the mean of these per-class APs.
    This is NOT a true mAP; it's a rough confidence-based proxy for display only.
    """
    if not running_scores_per_class:
        return None
    ap_values = []
    for cid, scores in running_scores_per_class.items():
        if len(scores) > 0:
            ap_values.append(float(np.mean(scores)))
    if not ap_values:
        return None
    return float(np.mean(ap_values))

# =========================
# Main Application
# =========================
def main():
    # Load labels
    labels = load_labels(label_path)
    # Setup interpreter
    interpreter = make_interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input tensor properties
    in_h, in_w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    in_dtype = input_details[0]['dtype']
    in_quant = input_details[0].get('quantization', None)

    # Video IO
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open input video: {input_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot open output video for writing: {output_path}")
        cap.release()
        return

    # For proxy mAP
    running_scores_per_class = {}

    frame_index = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_tensor = preprocess_frame(frame, (in_h, in_w), in_dtype, in_quant)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)

        # Inference
        interpreter.invoke()

        # Get outputs
        boxes, classes, scores, count = get_output_tensors(interpreter)
        if boxes is None or classes is None or scores is None:
            print("ERROR: Model outputs not found or in unexpected format.")
            break

        # Flatten outputs
        boxes = boxes[0] if boxes.ndim == 3 else boxes
        classes = classes[0] if classes.ndim == 2 else classes
        scores = scores[0] if scores.ndim == 2 else scores
        num_dets = int(count[0]) if count is not None else len(scores)

        # Collect detections above threshold
        detections = []
        for i in range(num_dets):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            class_id = int(classes[i])
            x1, y1, x2, y2 = scale_box_to_frame(boxes[i], orig_w, orig_h)
            detections.append({
                'class_id': class_id,
                'score': score,
                'box': (x1, y1, x2, y2),
            })
            # Update running scores for proxy mAP
            if class_id not in running_scores_per_class:
                running_scores_per_class[class_id] = []
            running_scores_per_class[class_id].append(score)

        # Draw detections
        draw_detections(frame, detections, labels)

        # Compute proxy mAP for overlay
        map_proxy = compute_proxy_map(running_scores_per_class)
        if map_proxy is not None:
            map_text = f"mAP (proxy): {map_proxy:.3f}"
        else:
            map_text = "mAP: N/A"

        # Overlay info: mAP and FPS
        elapsed = time.time() - t_start
        avg_fps = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
        info_text = f"{map_text} | FPS: {avg_fps:.2f}"
        (tw, th), bl = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 8, 10 + th + 14), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (14, 10 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Write frame
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    # Final log for proxy mAP
    final_map = compute_proxy_map(running_scores_per_class)
    if final_map is not None:
        print(f"Finished. Saved to: {output_path}")
        print(f"Proxy mAP (mean confidence across classes): {final_map:.4f}")
    else:
        print(f"Finished. Saved to: {output_path}")
        print("mAP could not be computed (no detections or no ground-truth provided).")

if __name__ == "__main__":
    main()