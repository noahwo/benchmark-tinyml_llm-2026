import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5
EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# =========================
# Utility Functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print(f"Warning: Label file not found at {path}. Using class IDs as labels.")
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Support two formats:
    # 1) "0 person"
    # 2) "person" (implicit index)
    for idx, line in enumerate(lines):
        if line.split(' ', 1)[0].isdigit():
            label_id, label_name = line.split(' ', 1)
            labels[int(label_id)] = label_name.strip()
        else:
            labels[idx] = line
    return labels

def make_interpreter(model_path):
    delegates = [load_delegate(EDGETPU_SHARED_LIB)]
    return Interpreter(model_path=model_path, experimental_delegates=delegates)

def set_input_tensor(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    input_dtype = input_details['dtype']
    # Ensure correct dtype and scale if needed
    if input_dtype == np.uint8:
        tensor = image.astype(np.uint8)
    else:
        # float32 model: normalize to [0,1]
        tensor = image.astype(np.float32) / 255.0
    interpreter.set_tensor(input_index, np.expand_dims(tensor, axis=0))

def get_output_tensors(interpreter):
    # Attempt to parse standard TFLite detection postprocess outputs
    output_details = interpreter.get_output_details()
    boxes = None
    classes = None
    scores = None
    num = None

    # Identify tensors by characteristics
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        shape = data.shape
        if len(shape) == 2 and shape[-1] == 4:
            # Some models might output [N,4], but most are [1, N, 4]
            # Normalize to [1, N, 4]
            boxes = np.expand_dims(data, axis=0)
        elif len(shape) == 3 and shape[-1] == 4:
            boxes = data
        elif len(shape) == 2:
            # Could be classes or scores; inspect dtype
            if data.dtype == np.float32:
                # Could be scores
                scores = np.expand_dims(data, axis=0)
            else:
                classes = np.expand_dims(data, axis=0)
        elif len(shape) == 3:
            # Could be classes or scores with [1, N, 1] or [1, N]
            if shape[-1] == 1:
                casted = data.squeeze(-1)
                if casted.dtype == np.float32:
                    scores = casted
                else:
                    classes = casted
            else:
                # [1, N] style
                if data.dtype == np.float32:
                    scores = data
                else:
                    classes = data
        elif len(shape) == 1 and shape[0] == 1:
            num = int(np.squeeze(data).astype(np.int32))

    # Fallback to standard order if still None
    if boxes is None or scores is None or classes is None:
        # Try standard indices: 0: boxes, 1: classes, 2: scores, 3: num
        try:
            boxes = interpreter.get_tensor(output_details[0]['index'])
            classes = interpreter.get_tensor(output_details[1]['index'])
            scores = interpreter.get_tensor(output_details[2]['index'])
            if len(output_details) > 3:
                num = int(np.squeeze(interpreter.get_tensor(output_details[3]['index'])).astype(np.int32))
        except Exception:
            # As a last resort, return empty results
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32), 0

    # Squeeze leading batch dim if present
    boxes = np.squeeze(boxes, axis=0) if boxes.ndim == 3 else boxes
    scores = np.squeeze(scores, axis=0) if scores.ndim == 3 else scores
    classes = np.squeeze(classes, axis=0) if classes.ndim == 3 else classes

    # Ensure shapes: [N,4], [N], [N]
    if boxes.ndim == 2 and boxes.shape[-1] == 4:
        N = boxes.shape[0]
        if scores.ndim == 1 and scores.shape[0] == N and classes.ndim == 1 and classes.shape[0] == N:
            pass
        else:
            # Attempt to squeeze
            scores = scores.reshape(-1)
            classes = classes.reshape(-1)
            N = min(N, scores.shape[0], classes.shape[0])
            boxes = boxes[:N]
            scores = scores[:N]
            classes = classes[:N]
    else:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32), 0

    if num is None:
        num = boxes.shape[0]
    return boxes, scores, classes.astype(np.int32), int(num)

def letterbox_image(image, target_size):
    # Not using letterbox for detection postprocess models; simple resize suffices.
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def preprocess_frame(frame, input_details):
    # Get expected input size
    _, in_h, in_w, in_c = input_details['shape']
    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    return resized

def generate_color_for_class(class_id):
    # Deterministic color for each class id
    np.random.seed(class_id)
    color = tuple(int(c) for c in np.random.randint(60, 255, size=3))
    # Convert to BGR for OpenCV
    return (color[2], color[1], color[0])

def draw_label(img, text, x, y, color=(255, 255, 255), scale=0.5, thickness=1):
    # Draw a filled rectangle with text on top-left at (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    # Ensure y is above the bbox if possible
    y = max(th + 2, y)
    # Background rectangle
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + 2), color, -1)
    # Put text in black for contrast
    cv2.putText(img, text, (x + 1, y - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

def clip_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2

def main():
    # Step 1. Setup: load interpreter with EdgeTPU delegate, allocate tensors, load labels, open video IO
    labels = load_labels(LABEL_PATH)

    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {INPUT_PATH}")
        return

    # Prepare video writer with same size as input frames
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0  # Fallback if metadata missing

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Stats and proxy mAP accumulator (mean confidence over kept detections)
    total_frames = 0
    total_inference_time = 0.0
    kept_detections_conf_sum = 0.0
    kept_detections_count = 0

    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        # Step 2. Preprocessing
        input_tensor = preprocess_frame(frame, input_details)

        # Step 3. Inference
        set_input_tensor(interpreter, input_tensor)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        total_inference_time += (t1 - t0)

        # Step 4. Output handling: decode, draw, compute proxy mAP, write
        boxes, scores, classes, num = get_output_tensors(interpreter)

        # Convert and draw detections
        for i in range(min(num, boxes.shape[0])):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue

            # Typical model returns [ymin, xmin, ymax, xmax] in normalized coords
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * frame_w)
            y1 = int(ymin * frame_h)
            x2 = int(xmax * frame_w)
            y2 = int(ymax * frame_h)
            x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, frame_w, frame_h)

            class_id = int(classes[i])
            class_name = labels.get(class_id, f"id:{class_id}")
            label_text = f"{class_name} {score:.2f}"

            color = generate_color_for_class(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Place label slightly above the top-left corner of the box
            draw_label(frame, label_text, x1, max(15, y1), color=color, scale=0.55, thickness=1)

            kept_detections_conf_sum += score
            kept_detections_count += 1

        # Compute and overlay proxy mAP (mean of confidences for kept detections)
        proxy_map = (kept_detections_conf_sum / kept_detections_count) if kept_detections_count > 0 else 0.0
        map_text = f"mAP: {proxy_map:.3f}"
        draw_label(frame, map_text, 10, 25, color=(255, 255, 255), scale=0.7, thickness=2)

        writer.write(frame)

    # Release resources
    cap.release()
    writer.release()

    # Summary
    avg_inf_ms = (total_inference_time / max(1, total_frames)) * 1000.0
    final_proxy_map = (kept_detections_conf_sum / kept_detections_count) if kept_detections_count > 0 else 0.0
    print("Processing complete.")
    print(f"Frames processed: {total_frames}")
    print(f"Average inference time: {avg_inf_ms:.2f} ms")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Proxy mAP (mean confidence over detections >= {CONFIDENCE_THRESHOLD}): {final_proxy_map:.3f}")

if __name__ == "__main__":
    main()