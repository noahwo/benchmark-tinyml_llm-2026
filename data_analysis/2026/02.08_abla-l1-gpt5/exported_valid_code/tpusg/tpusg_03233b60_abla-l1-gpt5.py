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
CONF_THRESHOLD = 0.5
EDGETPU_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(label_path):
    labels = {}
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try "id label" format first, else assume plain label per line
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
            else:
                labels[idx] = line
    return labels


def make_interpreter(model_path):
    return Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(EDGETPU_LIB)]
    )


def preprocess(frame_bgr, input_shape, input_dtype):
    # input_shape: [1, h, w, c]
    in_h, in_w = input_shape[1], input_shape[2]
    # Convert BGR to RGB for most TFLite models
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    else:
        inp = resized.astype(np.uint8)
    inp = np.expand_dims(inp, axis=0)
    return inp


def parse_detections(interpreter, output_details):
    # Most EdgeTPU SSD models output: boxes, classes, scores, num_detections
    out_tensors = [interpreter.get_tensor(od['index']) for od in output_details]
    # Attempt to map by common order
    # boxes: (1, N, 4), classes: (1, N), scores: (1, N), num: (1,)
    boxes = None
    classes = None
    scores = None
    num = None

    # First pass: try to use name hints
    name_map = {i: od.get('name', '') for i, od in enumerate(output_details)}
    for i, name in name_map.items():
        t = out_tensors[i]
        if 'PostProcess' in name and (t.ndim == 3 and t.shape[-1] == 4):
            boxes = t
        elif (':1' in name) or ('classes' in name.lower()):
            classes = t
        elif (':2' in name) or ('scores' in name.lower()):
            scores = t
        elif (':3' in name) or ('num' in name.lower()):
            num = t

    # Fallback mapping if any are None
    if boxes is None or classes is None or scores is None or num is None:
        candidates = out_tensors
        # boxes
        for t in candidates:
            if t.ndim == 3 and t.shape[-1] == 4:
                boxes = t
                break
        # scores and classes
        rest = [t for t in candidates if t is not boxes]
        # scores: float, shape (1, N)
        for t in rest:
            if t.ndim == 2 and t.shape[0] == 1 and t.dtype == np.float32:
                if scores is None:
                    scores = t
                else:
                    # whichever has more "score-like" values
                    if np.mean(scores) < np.mean(t):
                        scores = t
        # classes: same shape (1, N), float or int
        for t in rest:
            if t.ndim == 2 and t.shape[0] == 1 and t is not scores:
                classes = t
                break
        # num: shape (1,) or (1,1)
        for t in candidates:
            if t.size == 1:
                num = t
                break

    # Ensure shapes
    if boxes is None or classes is None or scores is None or num is None:
        return [], [], [], 0

    boxes = np.squeeze(boxes, axis=0)
    classes = np.squeeze(classes, axis=0)
    scores = np.squeeze(scores, axis=0)
    if num.ndim > 1:
        num = int(num.flatten()[0])
    else:
        num = int(num[0])
    num = min(num, boxes.shape[0], scores.shape[0], classes.shape[0])

    return boxes[:num], classes[:num], scores[:num], num


def denormalize_box(box, frame_w, frame_h):
    # box: [ymin, xmin, ymax, xmax] normalized
    ymin, xmin, ymax, xmax = box
    x1 = max(0, min(frame_w - 1, int(xmin * frame_w)))
    y1 = max(0, min(frame_h - 1, int(ymin * frame_h)))
    x2 = max(0, min(frame_w - 1, int(xmax * frame_w)))
    y2 = max(0, min(frame_h - 1, int(ymax * frame_h)))
    return x1, y1, x2, y2


def draw_text_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1, padding=3):
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # Ensure box fits within image
    x2 = min(img.shape[1] - 1, x + tw + 2 * padding)
    y2 = min(img.shape[0] - 1, y + th + base + 2 * padding)
    x1 = max(0, x)
    y1 = max(0, y - th - base - 2 * padding)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x + padding, y - base), font, font_scale, color, thickness, cv2.LINE_AA)


def main():
    # Validate paths
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU delegate
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    # Prepare output video writer
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # For mAP display: we don't have ground truth; show N/A
    running_map_value = None  # Will remain None unless ground truth is integrated by the user.

    frame_index = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Preprocess
            inp = preprocess(frame_bgr, input_shape, input_dtype)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], inp)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            infer_ms = (time.time() - t0) * 1000.0

            # Parse outputs
            boxes, classes, scores, num = parse_detections(interpreter, output_details)

            # Draw detections
            for i in range(num):
                score = float(scores[i])
                if score < CONF_THRESHOLD:
                    continue
                cls_id = int(classes[i])
                label = labels.get(cls_id, f"id:{cls_id}")
                x1, y1, x2, y2 = denormalize_box(boxes[i], frame_w, frame_h)
                color = ((cls_id * 123) % 255, (cls_id * 231) % 255, (cls_id * 97) % 255)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                caption = f"{label} {score:.2f}"
                text_y = y1 - 5 if y1 - 5 > 10 else y1 + 15
                draw_text_with_bg(frame_bgr, caption, (x1, text_y), font_scale=0.5, color=(255, 255, 255), bg_color=color, thickness=1)

            # Overlay inference time and mAP
            info_text = f"Infer: {infer_ms:.1f} ms"
            draw_text_with_bg(frame_bgr, info_text, (10, 25), font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1)

            map_text = "mAP: N/A"
            if running_map_value is not None:
                map_text = f"mAP: {running_map_value:.3f}"
            draw_text_with_bg(frame_bgr, map_text, (10, 50), font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1)

            # Write frame
            writer.write(frame_bgr)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    # Final message
    print(f"Processing complete. Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()