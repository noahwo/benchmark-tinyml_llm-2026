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
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"  # Read a single video file from the given input_path
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"  # Output the video file with rectangles drew on the detected objects, along with texts of labels and calculated mAP(mean average precision)
confidence_threshold = 0.5

# EdgeTPU shared library path on Google Coral Dev Board (aarch64)
EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(path):
    """
    Load labels from a text file.
    Supports both:
      - "index label" per line (e.g., "0 person")
      - "label" per line (index implied by line order starting at 0)
    """
    labels = {}
    try:
        with open(path, 'r') as f:
            idx = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                else:
                    labels[idx] = line
                    idx += 1
    except Exception as e:
        print(f"Failed to load labels from {path}: {e}")
        labels = {}
    return labels


def make_interpreter_with_edgetpu(model_file, delegate_lib):
    """
    Create a TFLite interpreter with EdgeTPU delegate.
    """
    try:
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=[load_delegate(delegate_lib)]
        )
        return interpreter
    except Exception as e:
        raise RuntimeError(f"Failed to create EdgeTPU interpreter: {e}")


def get_input_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    height = input_details['shape'][1]
    width = input_details['shape'][2]
    dtype = input_details['dtype']
    return width, height, dtype


def preprocess(frame_bgr, input_w, input_h, input_dtype):
    """
    Preprocess the input frame:
      - resize to model input size
      - convert BGR to RGB
      - convert dtype as required by the model
      - add batch dimension
    """
    resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.uint8:
        input_tensor = rgb.astype(np.uint8)
    else:
        # float input expected
        input_tensor = rgb.astype(np.float32) / 255.0

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def extract_detections(interpreter, frame_w, frame_h, conf_thresh):
    """
    Extract detections from interpreter outputs.
    Attempts to match typical TFLite detection postprocess tensors:
      - boxes: [1, N, 4] in normalized ymin, xmin, ymax, xmax
      - classes: [1, N] or [N]
      - scores: [1, N] or [N]
      - count: [1] number of valid detections
    Returns a list of dicts: { 'bbox': (x1,y1,x2,y2), 'class_id': int, 'score': float }
    """
    output_details = interpreter.get_output_details()
    tensors = {}
    for od in output_details:
        name = od.get('name', '')
        data = interpreter.get_tensor(od['index'])
        tensors[name] = data

    boxes = None
    classes = None
    scores = None
    count = None

    # Try assign by canonical names first
    for name, data in tensors.items():
        lname = name.lower()
        if 'box' in lname:
            boxes = data
        elif 'score' in lname:
            scores = data
        elif 'class' in lname:
            classes = data
        elif 'count' in lname or 'num_detections' in lname:
            count = data

    # Fallback by shapes if needed
    if boxes is None or scores is None or classes is None:
        # Collect all outputs
        outs = [interpreter.get_tensor(od['index']) for od in output_details]
        # Identify boxes (has last dim 4)
        for out in outs:
            if out.ndim >= 2 and out.shape[-1] == 4:
                boxes = out
                break
        # Identify scores (float, not 4 dims last)
        cand = []
        for out in outs:
            if out is boxes:
                continue
            if out.dtype in (np.float32, np.float64) and out.size > 1:
                cand.append(out)
        # Among candidates, one likely matches classes (float/int) and scores (float)
        # Choose the one with more unique fractional parts as scores
        # But to keep simple: assign the larger-magnitude as scores, the other as classes
        if len(cand) >= 2:
            a, b = cand[0], cand[1]
            if a.max() > b.max():
                scores, classes = a, b
            else:
                scores, classes = b, a
        elif len(cand) == 1:
            scores = cand[0]

        # Count (scalar)
        for out in outs:
            if out.size == 1 and (out.dtype == np.float32 or out.dtype == np.int32):
                count = out
                break

    # Normalize shapes to [N, ...]
    if boxes is not None and boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if classes is not None and classes.ndim == 2 and classes.shape[0] == 1:
        classes = classes[0]
    if scores is not None and scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]

    n = 0
    if count is not None:
        n = int(np.squeeze(count).astype(np.int32))
    else:
        if scores is not None:
            n = scores.shape[0]
        elif boxes is not None:
            n = boxes.shape[0]

    detections = []
    if boxes is None or scores is None or classes is None or n == 0:
        return detections

    # Determine if boxes are normalized [0,1]
    is_normalized = True
    try:
        if boxes.max() > 2.0:
            is_normalized = False
    except Exception:
        is_normalized = True

    for i in range(n):
        score = float(scores[i])
        if score < conf_thresh:
            continue
        cls_id = int(classes[i])

        box = boxes[i]
        if is_normalized:
            ymin = max(0.0, min(1.0, float(box[0])))
            xmin = max(0.0, min(1.0, float(box[1])))
            ymax = max(0.0, min(1.0, float(box[2])))
            xmax = max(0.0, min(1.0, float(box[3])))

            x1 = int(xmin * frame_w)
            y1 = int(ymin * frame_h)
            x2 = int(xmax * frame_w)
            y2 = int(ymax * frame_h)
        else:
            # Assume absolute pixel coordinates
            x1 = int(max(0, min(frame_w - 1, float(box[1]))))
            y1 = int(max(0, min(frame_h - 1, float(box[0]))))
            x2 = int(max(0, min(frame_w - 1, float(box[3]))))
            y2 = int(max(0, min(frame_h - 1, float(box[2]))))

        # Sanity check and clamp
        x1, x2 = max(0, min(x1, frame_w - 1)), max(0, min(x2, frame_w - 1))
        y1, y2 = max(0, min(y1, frame_h - 1)), max(0, min(y2, frame_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            'bbox': (x1, y1, x2, y2),
            'class_id': cls_id,
            'score': score
        })

    return detections


def color_for_class(class_id):
    # Deterministic pseudo-color based on class id
    r = int((37 * (class_id + 1)) % 255)
    g = int((17 * (class_id + 1)) % 255)
    b = int((29 * (class_id + 1)) % 255)
    return (b, g, r)


def main():
    # 1) Setup
    labels = load_labels(label_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("Initializing TFLite interpreter with EdgeTPU...")
    interpreter = make_interpreter_with_edgetpu(model_path, EDGETPU_SHARED_LIB)
    interpreter.allocate_tensors()
    in_w, in_h, in_dtype = get_input_details(interpreter)
    print(f"Model input: {in_w}x{in_h}, dtype={in_dtype}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0  # Fallback FPS

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (src_w, src_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    print(f"Input: {input_path} ({src_w}x{src_h} @ {fps:.2f} fps)")
    print(f"Output: {output_path}")

    # Statistics for an approximate "mAP" (proxy due to lack of ground truth)
    # We will compute per-class average of detection scores >= threshold,
    # and then the mean across classes that appeared at least once.
    per_class_scores = {}  # class_id -> list of scores

    # 2) Process frames
    frame_count = 0
    t0 = time.time()
    last_time = t0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_count += 1

            # 2) Preprocessing
            input_tensor = preprocess(frame_bgr, in_w, in_h, in_dtype)

            # 3) Inference
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            # 4) Output handling: detections, draw, compute "mAP (approx.)"
            detections = extract_detections(interpreter, src_w, src_h, confidence_threshold)

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cls_id = det['class_id']
                score = det['score']

                # Update per-class scores
                if cls_id not in per_class_scores:
                    per_class_scores[cls_id] = []
                per_class_scores[cls_id].append(score)

                # Draw bounding box and label
                color = color_for_class(cls_id)
                thickness = max(1, int(round(0.002 * (src_w + src_h))))
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

                label_name = labels.get(cls_id, f"id:{cls_id}")
                label_text = f"{label_name} {score:.2f}"
                # Text background
                (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                y_text = max(th + 4, y1 - 4)
                cv2.rectangle(frame_bgr, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), color, -1)
                cv2.putText(frame_bgr, label_text, (x1 + 2, y_text - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Compute approximate mAP (proxy): mean of per-class average scores
            map_approx = None
            if len(per_class_scores) > 0:
                class_means = []
                for cls_id, scores in per_class_scores.items():
                    if len(scores) > 0:
                        class_means.append(float(np.mean(np.array(scores, dtype=np.float32))))
                if len(class_means) > 0:
                    map_approx = float(np.mean(np.array(class_means, dtype=np.float32)))

            # Overlay performance info
            now = time.time()
            elapsed = now - last_time
            last_time = now
            inst_fps = 1.0 / elapsed if elapsed > 1e-6 else 0.0

            info_text = f"Detections: {len(detections)} | FPS: {inst_fps:.1f}"
            cv2.putText(frame_bgr, info_text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 240, 10), 2, cv2.LINE_AA)

            if map_approx is not None:
                cv2.putText(frame_bgr, f"mAP (approx): {map_approx:.3f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 240, 10), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_bgr, "mAP (approx): N/A", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 240, 10), 2, cv2.LINE_AA)

            writer.write(frame_bgr)

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    avg_fps = frame_count / total_time if total_time > 1e-6 else 0.0
    print(f"Processed {frame_count} frames in {total_time:.2f}s (avg FPS: {avg_fps:.2f})")
    if per_class_scores:
        class_means = [float(np.mean(np.array(s, dtype=np.float32))) for s in per_class_scores.values() if len(s) > 0]
        if class_means:
            map_approx = float(np.mean(np.array(class_means, dtype=np.float32)))
            print(f"Final mAP (approx): {map_approx:.4f}")
    print("Output saved to:", output_path)


if __name__ == "__main__":
    main()