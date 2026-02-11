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

EDGETPU_DELEGATE_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i, line in enumerate(lines):
        # Support "id: label" or "id label" or just "label"
        if ":" in line:
            idx_str, name = line.split(":", 1)
            idx_str = idx_str.strip()
            name = name.strip()
            try:
                idx = int(idx_str)
                labels[idx] = name
            except ValueError:
                # Fall back to index by line number
                labels[i] = line.strip()
        else:
            parts = line.split()
            if len(parts) > 1 and parts[0].isdigit():
                idx = int(parts[0])
                name = " ".join(parts[1:]).strip()
                labels[idx] = name
            else:
                labels[i] = line.strip()
    return labels


def make_interpreter(model_path):
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate(EDGETPU_DELEGATE_PATH)],
        )
    except ValueError as e:
        raise SystemExit(f"Failed to load EdgeTPU delegate: {e}")
    interpreter.allocate_tensors()
    return interpreter


def get_input_details(interpreter):
    input_details = interpreter.get_input_details()
    if not input_details:
        raise RuntimeError("No input details found in the TFLite model.")
    det = input_details[0]
    shape = det["shape"]
    dtype = det["dtype"]
    # Expect shape [1, height, width, 3]
    height, width = int(shape[1]), int(shape[2])
    quant = det.get("quantization", (0.0, 0))
    return width, height, dtype, quant, det["index"]


def get_output_tensors(interpreter):
    # Attempt to extract boxes, classes, scores, num_detections from outputs
    output_details = interpreter.get_output_details()
    boxes = None
    classes = None
    scores = None
    num = None
    for od in output_details:
        out = interpreter.get_tensor(od["index"])
        out_squeezed = np.squeeze(out)
        shp = out_squeezed.shape
        # Identify by shape heuristics
        if len(shp) == 2 and shp[1] == 4:
            boxes = out_squeezed  # (N, 4)
        elif len(shp) == 1:
            # Could be classes, scores, or num
            if out_squeezed.dtype in (np.float32, np.float64):
                # Could be scores (float)
                if scores is None or out_squeezed.size > scores.size:
                    scores = out_squeezed
            elif np.issubdtype(out_squeezed.dtype, np.integer):
                # Could be classes (int) or num (int)
                # Typically classes are float but sometimes int
                if out_squeezed.size == 1:
                    num = int(out_squeezed.item())
                else:
                    classes = out_squeezed.astype(np.int32)
        elif len(shp) == 2 and (out_squeezed.dtype in (np.float32, np.float64)):
            # Could be boxes (N,4) already captured, or classes/scores if different shapes
            if shp[1] == 4:
                boxes = out_squeezed
            else:
                # If scores hasn't been set, assume this is scores
                if scores is None:
                    scores = out_squeezed[:, 0] if shp[1] == 1 else np.max(out_squeezed, axis=1)
        # Handle typical case of batched outputs (1, N, 4) etc.
        if len(np.shape(out)) == 3 and np.shape(out)[-1] == 4:
            boxes = np.squeeze(out, axis=0)
        if len(np.shape(out)) == 2 and np.shape(out)[0] == 1 and np.issubdtype(out.dtype, np.floating):
            # Could be classes or scores
            arr = np.squeeze(out, axis=0)
            # If values are <=1 it's more likely scores, else classes (depending on model)
            if np.all((arr >= 0) & (arr <= 1.0)):
                scores = arr
            else:
                # classes can be float
                classes = arr.astype(np.int32)
        if len(np.shape(out)) == 1 and np.shape(out)[0] == 1:
            # num detections
            try:
                num = int(out.flatten()[0])
            except Exception:
                pass

    # Sanity fallback, if num not provided, infer from boxes or scores shape
    if num is None:
        if boxes is not None:
            num = boxes.shape[0]
        elif scores is not None:
            num = scores.shape[0]
        elif classes is not None:
            num = classes.shape[0]
        else:
            num = 0

    # Align sizes
    if boxes is not None and boxes.shape[0] > num:
        boxes = boxes[:num]
    if scores is not None and scores.shape[0] > num:
        scores = scores[:num]
    if classes is not None and classes.shape[0] > num:
        classes = classes[:num]

    return boxes, classes, scores, num


def preprocess_frame(frame_bgr, input_w, input_h, input_dtype):
    # Convert BGR to RGB and resize to model input size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    if input_dtype == np.float32:
        # Normalize to [0,1]
        input_data = resized.astype(np.float32) / 255.0
    else:
        # Assume uint8 quantized model expects [0,255] uint8
        input_data = resized.astype(np.uint8)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def draw_detections(frame_bgr, detections, labels, mAP_value=None, fps_text=None):
    # Draw bounding boxes and labels
    h, w = frame_bgr.shape[:2]
    for det in detections:
        ymin, xmin, ymax, xmax, score, class_id = det
        # Convert normalized coordinates to absolute pixel positions
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        # Choose color based on class
        color = ((37 * (class_id + 1)) % 255, (17 * (class_id + 7)) % 255, (29 * (class_id + 11)) % 255)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label_text = labels.get(class_id, str(class_id))
        caption = f"{label_text}: {score:.2f}"
        cv2.putText(frame_bgr, caption, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Overlay mAP (proxy) if provided
    y_offset = 20
    if mAP_value is not None:
        cv2.putText(frame_bgr, f"mAP (proxy): {mAP_value:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 24

    # Overlay FPS if provided
    if fps_text:
        cv2.putText(frame_bgr, fps_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr


def compute_map_proxy(per_class_scores):
    # Proxy mAP: average of per-class mean confidence scores (no ground truth available)
    # This is NOT a true mAP; used here as a lightweight proxy since GT is not provided.
    ap_values = []
    for cls_id, scores in per_class_scores.items():
        if len(scores) > 0:
            ap_values.append(float(np.mean(scores)))
    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))


def main():
    # Validate paths
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(LABEL_PATH):
        raise SystemExit(f"Label file not found: {LABEL_PATH}")
    if not os.path.isfile(INPUT_PATH):
        raise SystemExit(f"Input video file not found: {INPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading labels...")
    labels = load_labels(LABEL_PATH)

    print("Initializing TFLite interpreter with EdgeTPU delegate...")
    interpreter = make_interpreter(MODEL_PATH)
    input_w, input_h, input_dtype, input_quant, input_index = get_input_details(interpreter)
    print(f"Model input: {input_w}x{input_h}, dtype={input_dtype}")

    print(f"Opening input video: {INPUT_PATH}")
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open input video: {INPUT_PATH}")

    # Get input video properties
    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_width, in_height))
    if not out_writer.isOpened():
        raise SystemExit(f"Failed to create output video: {OUTPUT_PATH}")

    print(f"Writing annotated output to: {OUTPUT_PATH}")
    print("Processing frames...")

    per_class_scores = {}  # class_id -> list of confidences
    frame_count = 0
    t_infer_total = 0.0

    last_time = time.time()
    fps_smooth = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1

        # Preprocess
        input_data = preprocess_frame(frame_bgr, input_w, input_h, input_dtype)

        # Inference
        interpreter.set_tensor(input_index, input_data)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000.0
        t_infer_total += (t1 - t0)

        # Postprocess: extract detections
        boxes, classes, scores, num = get_output_tensors(interpreter)

        detections = []
        if boxes is not None and scores is not None and classes is not None:
            N = min(num, boxes.shape[0], scores.shape[0], classes.shape[0])
            for i in range(N):
                score = float(scores[i])
                if score < CONFIDENCE_THRESHOLD:
                    continue
                ymin, xmin, ymax, xmax = boxes[i].tolist()
                class_id = int(classes[i])
                detections.append((ymin, xmin, ymax, xmax, score, class_id))
                # Update proxy-mAP stats
                if class_id not in per_class_scores:
                    per_class_scores[class_id] = []
                per_class_scores[class_id].append(score)

        # Compute proxy mAP to overlay in current frame
        map_proxy = compute_map_proxy(per_class_scores)

        # Smooth FPS display
        now = time.time()
        dt = now - last_time
        last_time = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        if fps_smooth is None:
            fps_smooth = inst_fps
        else:
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
        fps_text = f"FPS: {fps_smooth:.1f}  Infer: {infer_ms:.1f} ms"

        # Draw results on frame
        annotated = draw_detections(frame_bgr.copy(), detections, labels, mAP_value=map_proxy, fps_text=fps_text)

        # Write to output video
        out_writer.write(annotated)

    cap.release()
    out_writer.release()

    overall_map_proxy = compute_map_proxy(per_class_scores)
    avg_infer_ms = (t_infer_total / max(1, frame_count)) * 1000.0

    print("Processing complete.")
    print(f"Frames processed: {frame_count}")
    print(f"Average inference time: {avg_infer_ms:.2f} ms")
    print(f"Proxy mAP over video (no GT provided): {overall_map_proxy:.3f}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()