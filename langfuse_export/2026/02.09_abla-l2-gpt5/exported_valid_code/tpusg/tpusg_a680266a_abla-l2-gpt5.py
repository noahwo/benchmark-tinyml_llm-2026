import os
import time
import numpy as np
import cv2

# TFLite/EdgeTPU
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError as e:
    raise SystemExit("tflite_runtime is required on the Coral Dev Board. Install it and retry.") from e

# ----------------------------
# CONFIGURATION PARAMETERS
# ----------------------------
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(path):
    """
    Loads labels from a text file into a dict of {int_id: label_str}.
    Supports formats:
      - One label per line (implicit 0-based indexing)
      - "id label" pairs per line (space or comma separated)
    """
    labels = {}
    if not os.path.exists(path):
        print(f"Label file not found: {path}")
        return labels

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Detect format
    indexed = True
    for line in lines:
        parts = line.replace(",", " ").split()
        if len(parts) < 2 or not parts[0].isdigit():
            indexed = False
            break

    if indexed:
        for line in lines:
            parts = line.replace(",", " ").split()
            try:
                idx = int(parts[0])
                name = " ".join(parts[1:]).strip()
                labels[idx] = name
            except Exception:
                continue
    else:
        for i, line in enumerate(lines):
            labels[i] = line

    return labels


def get_label(labels, class_id):
    """
    Safely return a label string for the given class_id.
    Tries exact id, then id+1 (to handle common 1-based label maps), else str(class_id).
    """
    if class_id in labels:
        return labels[class_id]
    if (class_id + 1) in labels:
        return labels[class_id + 1]
    return str(class_id)


def make_interpreter(model_path, edgetpu_lib):
    """
    Creates a TFLite interpreter with EdgeTPU delegate.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        delegate = load_delegate(edgetpu_lib)
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
    except Exception as e:
        raise SystemExit(f"Failed to load EdgeTPU delegate from {edgetpu_lib}: {e}")
    interpreter.allocate_tensors()
    return interpreter


def prepare_input(frame_bgr, input_details):
    """
    Preprocesses a frame for the model:
      - Resize to input size
      - Convert BGR -> RGB
      - Convert dtype as needed (uint8 or float32 [0,1])
      - Add batch dimension
    Returns:
      input_tensor (np.ndarray), resized_frame_rgb
    """
    in_shape = input_details[0]["shape"]  # [1, h, w, c]
    in_h, in_w = int(in_shape[1]), int(in_shape[2])
    in_dtype = input_details[0]["dtype"]

    resized = cv2.resize(frame_bgr, (in_w, in_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if in_dtype == np.float32:
        input_tensor = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
    else:
        input_tensor = np.expand_dims(rgb.astype(in_dtype), axis=0)

    return input_tensor, rgb


def set_input_tensor(interpreter, input_tensor):
    """
    Copies input_tensor into the interpreter input buffer.
    """
    input_details = interpreter.get_input_details()
    index = input_details[0]["index"]
    interpreter.set_tensor(index, input_tensor)


def get_output_tensors(interpreter):
    """
    Retrieves and normalizes typical TFLite detection model outputs:
      boxes: [N, 4] in normalized ymin, xmin, ymax, xmax
      classes: [N] int
      scores: [N] float
      count: int
    Handles common variations in output order/dtypes.
    """
    output_details = interpreter.get_output_details()
    tensors = [interpreter.get_tensor(od["index"]) for od in output_details]

    # Flatten any leading batch dim of 1
    flat = []
    for t in tensors:
        tt = t
        while tt.ndim > 1 and tt.shape[0] == 1:
            tt = np.squeeze(tt, axis=0)
        flat.append(tt)

    boxes, classes, scores, count = None, None, None, None

    # Heuristic assignment by shape/dtype
    for arr in flat:
        if arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = arr.astype(np.float32)
        elif arr.ndim == 1 and arr.dtype.kind in ("i", "u"):
            classes = arr.astype(np.int32)
        elif arr.ndim == 1 and arr.dtype.kind == "f":
            # Could be scores or sometimes boxes if malformed; prefer 1D floats as scores
            if scores is None:
                scores = arr.astype(np.float32)
        elif arr.ndim == 0:
            count = int(arr)

    # Fallback for common 4-output order [boxes, classes, scores, count]
    if boxes is None or classes is None or scores is None or count is None:
        try:
            od = interpreter.get_output_details()
            b = interpreter.get_tensor(od[0]["index"]).squeeze(axis=0)
            c = interpreter.get_tensor(od[1]["index"]).squeeze(axis=0)
            s = interpreter.get_tensor(od[2]["index"]).squeeze(axis=0)
            n = interpreter.get_tensor(od[3]["index"]).squeeze()
            if b.ndim == 2 and b.shape[-1] == 4:
                boxes = b.astype(np.float32)
                # Sometimes classes/scores are swapped in dtype; fix if needed
                if c.dtype.kind == "f" and s.dtype.kind in ("i", "u"):
                    c, s = s, c
                classes = c.astype(np.int32)
                scores = s.astype(np.float32)
                count = int(n)
        except Exception:
            pass

    if boxes is None or classes is None or scores is None or count is None:
        raise RuntimeError("Unable to parse model outputs for detection.")

    # Ensure lengths align with count
    n = min(count, len(scores), len(classes), len(boxes))
    return boxes[:n], classes[:n], scores[:n], n


def draw_detections(frame_bgr, detections, labels, running_map_proxy):
    """
    Draw detection boxes and labels onto the frame.
    detections: list of (ymin, xmin, ymax, xmax, class_id, score) in absolute pixel coords.
    """
    h, w = frame_bgr.shape[:2]
    for (ymin, xmin, ymax, xmax, cid, score) in detections:
        # Clip to frame
        xmin = max(0, min(w - 1, int(xmin)))
        xmax = max(0, min(w - 1, int(xmax)))
        ymin = max(0, min(h - 1, int(ymin)))
        ymax = max(0, min(h - 1, int(ymax)))

        color = (0, 255, 0)  # Green boxes
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, 2)

        label = get_label(labels, int(cid))
        text = f"{label}: {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, ymin), color, -1)
        cv2.putText(frame_bgr, text, (xmin + 2, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay "mAP" proxy on the top-left
    map_text = f"mAP (proxy): {running_map_proxy:.3f}"
    (tw, th), bl = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (5, 5), (10 + tw, 10 + th), (255, 255, 255), -1)
    cv2.putText(frame_bgr, map_text, (8, 8 + th - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return frame_bgr


def main():
    # 1. Setup: interpreter, labels, video IO
    print("Initializing interpreter with EdgeTPU...")
    interpreter = make_interpreter(model_path, EDGETPU_LIB)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Interpreter initialized.")

    labels = load_labels(label_path)
    if not labels:
        print("Warning: Labels could not be loaded or label file is empty. Class IDs will be displayed.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open input video: {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 0.0 or np.isnan(in_fps):
        in_fps = 30.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, in_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open output video for writing: {output_path}")

    # Stats for "mAP" proxy and performance
    total_conf_sum = 0.0
    total_det_count = 0
    frame_idx = 0
    t0 = time.time()

    print("Processing video...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 2. Preprocessing
        input_tensor, _ = prepare_input(frame_bgr, input_details)

        # 3. Inference
        set_input_tensor(interpreter, input_tensor)
        interpreter.invoke()

        # 4. Output handling
        boxes, classes, scores, count = get_output_tensors(interpreter)

        detections = []
        # Convert normalized boxes to absolute pixel coords and filter by confidence
        for i in range(count):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            ymin_abs = ymin * height
            xmin_abs = xmin * width
            ymax_abs = ymax * height
            xmax_abs = xmax * width
            detections.append((ymin_abs, xmin_abs, ymax_abs, xmax_abs, int(classes[i]), score))

        # Update proxy "mAP" as mean confidence across all detections so far
        if len(detections) > 0:
            confs = [d[5] for d in detections]
            total_conf_sum += float(np.sum(confs))
            total_det_count += len(confs)
        running_map_proxy = (total_conf_sum / total_det_count) if total_det_count > 0 else 0.0

        # Draw and write frame
        annotated = draw_detections(frame_bgr.copy(), detections, labels, running_map_proxy)
        writer.write(annotated)

        # Optional: print simple progress
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / max(1e-6, elapsed)
            print(f"Processed {frame_idx} frames | Running FPS: {fps:.2f} | Detections so far: {total_det_count} | mAP (proxy): {running_map_proxy:.3f}")

    # Cleanup
    cap.release()
    writer.release()

    total_time = time.time() - t0
    overall_fps = frame_idx / max(1e-6, total_time)
    final_map_proxy = (total_conf_sum / total_det_count) if total_det_count > 0 else 0.0

    print("--------------------------------------------------")
    print(f"Finished processing {frame_idx} frames")
    print(f"Output saved to: {output_path}")
    print(f"Average processing FPS: {overall_fps:.2f}")
    print(f"Final mAP (proxy): {final_map_proxy:.4f}")
    print("Note: This 'mAP (proxy)' is computed as the mean of detection confidences across all detections,")
    print("      since ground-truth annotations are not provided for true mAP computation.")


if __name__ == "__main__":
    main()