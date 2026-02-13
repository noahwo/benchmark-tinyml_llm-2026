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


def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Supported formats:
            #  - "0: person"
            #  - "0 person"
            #  - "person" (index inferred by line number)
            #  - "item { id: 1 name: 'person' }" (best effort)
            if "item" in line and "id:" in line and "name:" in line:
                # Best-effort parsing of TF object_detection labelmap.pbtxt line (single-line)
                try:
                    parts = line.replace("{", " ").replace("}", " ").replace('"', "'").split()
                    id_idx = parts.index("id:") + 1
                    name_idx = parts.index("name:") + 1
                    cls_id = int(parts[id_idx])
                    name = parts[name_idx].strip("'").strip('"')
                    labels[cls_id] = name
                except Exception:
                    continue
            else:
                if ":" in line:
                    left, right = line.split(":", 1)
                    try:
                        cls_id = int(left.strip())
                        labels[cls_id] = right.strip()
                        continue
                    except ValueError:
                        pass
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    cls_id = int(parts[0])
                    labels[cls_id] = " ".join(parts[1:]).strip()
                else:
                    # Fallback: index by line number if no explicit id found
                    labels[i] = line
    return labels


def make_interpreter(model_path):
    delegates = [load_delegate(EDGETPU_SHARED_LIB)]
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    return interpreter


def get_output_tensors(interpreter):
    # Returns a dict with keys: 'boxes', 'classes', 'scores', 'num'
    details = interpreter.get_output_details()
    out = {"boxes": None, "classes": None, "scores": None, "num": None}
    for d in details:
        shape = d["shape"]
        name = d.get("name", "").lower()
        if len(shape) == 3 and shape[-1] == 4:
            out["boxes"] = d
        elif len(shape) == 2:
            # Could be classes or scores
            # Heuristics via name or dtype
            if "class" in name:
                out["classes"] = d
            elif "score" in name:
                out["scores"] = d
            else:
                # If not informative, decide later based on dtype
                if out["scores"] is None and np.dtype(d["dtype"]) in (np.float32, np.float64):
                    out["scores"] = d
                elif out["classes"] is None:
                    out["classes"] = d
        elif len(shape) == 1 and shape[0] == 1:
            out["num"] = d
    # Final sanity: all must be present
    if any(v is None for v in out.values()):
        # Attempt fallback by ordering: typical order is [boxes, classes, scores, num]
        if len(details) >= 4:
            out["boxes"] = details[0]
            out["classes"] = details[1]
            out["scores"] = details[2]
            out["num"] = details[3]
    return out


def preprocess_frame(frame_bgr, input_size, input_dtype):
    # Convert BGR to RGB and resize to model input size
    h_in, w_in = input_size
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (w_in, h_in))
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def draw_detections(frame, dets, labels):
    # dets: list of tuples (ymin, xmin, ymax, xmax, class_id, score)
    for (ymin, xmin, ymax, xmax, class_id, score) in dets:
        # Draw rectangle
        color = (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Prepare label text
        label = labels.get(class_id, str(class_id))
        text = f"{label}: {score:.2f}"

        # Text background
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, ymin), color, -1)
        cv2.putText(frame, text, (xmin + 2, max(0, ymin - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def compute_map_placeholder():
    # Without ground-truth annotations, true mAP cannot be computed.
    # This function exists to conform to the output requirement and returns None.
    return None


def main():
    # Setup: Interpreter with EdgeTPU
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    interpreter = make_interpreter(MODEL_PATH)

    # Input/output tensor details
    input_details = interpreter.get_input_details()
    output_map = get_output_tensors(interpreter)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Open video input
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Prepare video writer for output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_width, in_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # Determine model input shape
    # Expecting [1, height, width, 3]
    in_shape = input_details[0]["shape"]
    model_h, model_w = int(in_shape[1]), int(in_shape[2])
    input_dtype = input_details[0]["dtype"]

    frame_index = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing
        input_tensor = preprocess_frame(frame, (model_h, model_w), input_dtype)

        # Inference
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        # Collect outputs
        boxes = interpreter.get_tensor(output_map["boxes"]["index"])
        classes = interpreter.get_tensor(output_map["classes"]["index"])
        scores = interpreter.get_tensor(output_map["scores"]["index"])
        num = interpreter.get_tensor(output_map["num"]["index"])

        # Squeeze to 1D lists
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        num_det = int(np.squeeze(num))

        # Postprocessing: scale boxes to original image size and filter by confidence threshold
        detections = []
        for i in range(num_det):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(classes[i])
            ymin, xmin, ymax, xmax = boxes[i]

            # Clamp to [0,1]
            ymin = max(0.0, min(1.0, float(ymin)))
            xmin = max(0.0, min(1.0, float(xmin)))
            ymax = max(0.0, min(1.0, float(ymax)))
            xmax = max(0.0, min(1.0, float(xmax)))

            # Scale to original frame size
            x1 = int(xmin * in_width)
            y1 = int(ymin * in_height)
            x2 = int(xmax * in_width)
            y2 = int(ymax * in_height)

            # Ensure valid coordinates
            x1 = max(0, min(in_width - 1, x1))
            x2 = max(0, min(in_width - 1, x2))
            y1 = max(0, min(in_height - 1, y1))
            y2 = max(0, min(in_height - 1, y2))

            detections.append((y1, x1, y2, x2, cls_id, score))

        # Draw detections
        output_frame = draw_detections(frame.copy(), detections, labels)

        # Calculate mAP (placeholder, no GT available)
        current_map = compute_map_placeholder()
        map_text = "mAP: N/A (no ground truth)"
        if current_map is not None:
            map_text = f"mAP: {current_map:.3f}"

        # Overlay mAP and FPS info
        elapsed = time.time() - t0
        fps_runtime = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
        info_text = f"{map_text} | FPS: {fps_runtime:.1f}"
        (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_frame, (5, 5), (5 + tw + 8, 5 + th + 12), (50, 50, 50), -1)
        cv2.putText(output_frame, info_text, (9, 5 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Write output frame
        writer.write(output_frame)

        frame_index += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()