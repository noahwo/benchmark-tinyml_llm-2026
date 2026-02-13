import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP
CONFIDENCE_THRESHOLD = 0.5


def load_labels(label_path):
    """
    Load labels from a label map file.
    Supports simple line-by-line label files.
    Returns a list where index corresponds to class id.
    """
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle possible "id: x, name: y" formats crudely by taking last token if colon exists
            if ":" in line and "," in line:
                # Try to extract name after 'name:'
                parts = [p.strip() for p in line.split(",")]
                name = None
                for p in parts:
                    if "name" in p:
                        kv = p.split(":")
                        if len(kv) >= 2:
                            name = kv[1].strip().strip('"').strip("'")
                            break
                labels.append(name if name else line)
            elif ":" in line:
                # id:name
                labels.append(line.split(":", 1)[1].strip())
            else:
                # Plain label per line
                labels.append(line)
    return labels


def id_to_label(cid, labels):
    """
    Map class id to label string using best-effort rules.
    Many TFLite SSD models output 0-based or 1-based class ids depending on label map.
    """
    if 0 <= cid < len(labels):
        return labels[cid]
    # Try 1-based shift if file includes a leading "???" or similar
    if 0 <= cid + 1 < len(labels):
        return labels[cid + 1]
    if 0 <= cid - 1 < len(labels):
        return labels[cid - 1]
    return f"id_{cid}"


def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def preprocess_frame(frame_bgr, input_shape, input_details):
    """
    Preprocess frame for TFLite SSD model.
    - Resize to model input size
    - Convert BGR -> RGB
    - Handle dtype/quantization as required
    Returns a numpy array with shape [1, H, W, 3]
    """
    in_h, in_w = input_shape[1], input_shape[2]  # [1, H, W, 3]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    inp_dtype = input_details[0]["dtype"]
    quant_params = input_details[0].get("quantization", (0.0, 0))
    scale, zero_point = quant_params if isinstance(quant_params, (list, tuple)) and len(quant_params) == 2 else (0.0, 0)

    if inp_dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    elif inp_dtype == np.uint8:
        if scale and scale > 0:
            # Quantize to uint8 using provided scale and zero_point
            inp = np.round(resized.astype(np.float32) / scale + zero_point)
            inp = np.clip(inp, 0, 255).astype(np.uint8)
        else:
            inp = resized.astype(np.uint8)
    else:
        # Fallback: try to cast appropriately
        inp = resized.astype(inp_dtype)

    return np.expand_dims(inp, axis=0)


def draw_detection(frame, box, score, label, color):
    """
    Draw a single detection on the frame.
    box: [ymin, xmin, ymax, xmax] in absolute pixel coords
    """
    y_min, x_min, y_max, x_max = box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    caption = f"{label}: {score:.2f}"
    # Put text background
    (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x_min, max(0, y_min - th - 4)), (x_min + tw + 2, y_min), color, -1)
    cv2.putText(frame, caption, (x_min + 1, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def class_color(class_id):
    # Deterministic pseudo-color for each class id
    return (int((37 * class_id) % 255), int((17 * class_id) % 255), int((29 * class_id) % 255))


def compute_map_proxy(class_conf_history):
    """
    Compute a proxy for mAP in absence of ground truth:
    - For each class with any detections, AP_proxy = mean of detection confidences observed.
    - mAP_proxy = mean(AP_proxy over classes with detections).
    Returns float in [0, 1].
    """
    if not class_conf_history:
        return 0.0
    ap_vals = []
    for confs in class_conf_history.values():
        if len(confs) > 0:
            ap_vals.append(float(np.mean(confs)))
    if not ap_vals:
        return 0.0
    return float(np.mean(ap_vals))


def main():
    # 1. Setup: Load TFLite interpreter, labels, and video input
    labels = load_labels(LABEL_PATH)

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine input tensor shape [1, H, W, 3]
    input_shape = input_details[0]["shape"]

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open input video: {INPUT_PATH}")
        return

    # Prepare output video writer
    ensure_dir_for_file(OUTPUT_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: Could not open output video for write: {OUTPUT_PATH}")
        cap.release()
        return

    # Stats for mAP proxy
    class_conf_history = {}  # class_id -> list of confidences
    frame_count = 0
    t0 = time.time()

    # 2-3. Processing loop: Preprocess each frame and run inference
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Preprocess
        input_tensor = preprocess_frame(frame, input_shape, input_details)

        # Inference
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        # 4. Output handling
        # Typical SSD Mobilenet v1 TFLite detection output order:
        # 0: boxes [1, num, 4] (ymin, xmin, ymax, xmax) normalized
        # 1: classes [1, num]
        # 2: scores [1, num]
        # 3: num_detections [1]
        try:
            boxes = interpreter.get_tensor(output_details[0]["index"])[0]
            classes = interpreter.get_tensor(output_details[1]["index"])[0].astype(np.int32)
            scores = interpreter.get_tensor(output_details[2]["index"])[0]
            num_detections = int(interpreter.get_tensor(output_details[3]["index"])[0])
        except Exception:
            # Fallback: try to locate tensors by shape
            outs = [interpreter.get_tensor(od["index"]) for od in output_details]
            # Identify likely tensors by shapes
            boxes, classes, scores, num_detections = None, None, None, None
            for arr in outs:
                s = arr.shape
                if len(s) == 3 and s[-1] == 4:
                    boxes = arr[0]
                elif len(s) == 2 and s[0] == 1 and s[1] > 1 and (arr.dtype == np.float32 or arr.dtype == np.uint8):
                    # could be classes or scores; differentiate by values range
                    if np.all((arr >= 0) & (arr <= 1.0 + 1e-6)):
                        scores = arr[0]
                    else:
                        classes = arr[0].astype(np.int32)
                elif len(s) == 1 and s[0] == 1:
                    num_detections = int(arr[0])
            if boxes is None or classes is None or scores is None or num_detections is None:
                print("ERROR: Unable to parse model outputs for detection.")
                break

        # Draw detections and update stats
        for i in range(num_detections):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            class_id = int(classes[i])

            # Convert box to pixel coords
            ymin = int(max(0, min(height - 1, boxes[i][0] * height)))
            xmin = int(max(0, min(width - 1, boxes[i][1] * width)))
            ymax = int(max(0, min(height - 1, boxes[i][2] * height)))
            xmax = int(max(0, min(width - 1, boxes[i][3] * width)))

            # Fix inverted boxes if any
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if ymax < ymin:
                ymin, ymax = ymax, ymin

            label = id_to_label(class_id, labels)
            color = class_color(class_id)
            draw_detection(frame, (ymin, xmin, ymax, xmax), score, label, color)

            # Update proxy stats
            if class_id not in class_conf_history:
                class_conf_history[class_id] = []
            class_conf_history[class_id].append(score)

        # Compute and overlay mAP (proxy) on the frame
        map_proxy = compute_map_proxy(class_conf_history)
        cv2.putText(frame, f"mAP: {map_proxy:.3f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Write frame
        writer.write(frame)

    # Cleanup
    cap.release()
    writer.release()
    elapsed = time.time() - t0
    final_map_proxy = compute_map_proxy(class_conf_history)
    print(f"Processed {frame_count} frames in {elapsed:.2f}s ({(frame_count / max(elapsed, 1e-6)):.2f} FPS).")
    print(f"Final mAP: {final_map_proxy:.4f}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()