import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# CONFIGURATION PARAMETERS
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"  # Read a single video file from the given input_path
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP text
confidence_threshold = 0.5


def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Support two common formats:
            # 1) "0 person"
            # 2) "person"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[i] = line
    return labels


def make_interpreter(model_path_):
    # Initialize Interpreter with EdgeTPU delegate
    delegate_path = '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'
    interpreter_ = Interpreter(
        model_path=model_path_,
        experimental_delegates=[load_delegate(delegate_path)]
    )
    interpreter_.allocate_tensors()
    return interpreter_


def preprocess_frame(frame, input_details):
    # Get expected input size and dtype
    _, in_h, in_w, in_c = input_details['shape']
    dtype = input_details['dtype']

    # Resize and convert to RGB
    frame_resized = cv2.resize(frame, (in_w, in_h))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Expand dims to [1, h, w, c]
    if dtype == np.float32:
        # Normalize to [0,1] by default for float models
        input_data = np.expand_dims(frame_rgb.astype(np.float32) / 255.0, axis=0)
    else:
        # Quantized model expects uint8
        input_data = np.expand_dims(frame_rgb.astype(np.uint8), axis=0)
    return input_data


def set_input_tensor(interpreter_, input_data):
    input_details = interpreter_.get_input_details()[0]
    interpreter_.set_tensor(input_details['index'], input_data)


def get_output_tensors(interpreter_):
    # Typical TFLite SSD detection model returns:
    # boxes: [1, num, 4] (ymin, xmin, ymax, xmax) normalized
    # classes: [1, num]
    # scores: [1, num]
    # num_detections: [1]
    output_details = interpreter_.get_output_details()

    def data_at(idx):
        return interpreter_.get_tensor(output_details[idx]['index'])

    # Try to infer which tensor corresponds to what
    boxes = classes = scores = num = None
    for od in output_details:
        data = interpreter_.get_tensor(od['index'])
        shape = data.shape
        if len(shape) == 3 and shape[-1] == 4:
            boxes = data
        elif len(shape) == 2:
            # could be classes or scores
            if data.dtype == np.float32 and np.max(data) <= 1.0:
                # likely scores
                scores = data
            else:
                classes = data
        elif len(shape) == 1 and shape[0] == 1:
            num = data
        elif len(shape) == 2 and shape == (1, 1):
            num = data

    # Fallbacks if names/shapes are unexpected
    # Attempt to map by heuristics if any are None
    if boxes is None or classes is None or scores is None:
        # Try by indexing assumptions: [boxes, classes, scores, num]
        # This block is a last resort and may vary by model
        tensors = [interpreter_.get_tensor(od['index']) for od in output_details]
        for t in tensors:
            if t.ndim == 3 and t.shape[-1] == 4:
                boxes = t
        float_tensors = [t for t in tensors if t.dtype == np.float32]
        if scores is None:
            for t in float_tensors:
                if t.ndim == 2 and t.shape[0] == 1 and np.max(t) <= 1.0:
                    scores = t
                    break
        if classes is None:
            for t in tensors:
                if t.ndim == 2 and t.shape[0] == 1 and t is not scores:
                    classes = t
                    break
        if num is None:
            for t in tensors:
                if t.ndim in (1, 2) and np.size(t) == 1:
                    num = t
                    break

    return boxes, classes, scores, num


def draw_detections(frame, detections, labels, map_value):
    # Draw detection boxes and labels
    for det in detections:
        (ymin, xmin, ymax, xmax) = det['box']  # pixel coords
        cls_id = det['class_id']
        score = det['score']
        label = labels.get(cls_id, f'id {cls_id}')
        caption = f'{label}: {int(score * 100)}%'

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Text background
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        th = th + baseline
        y_text = max(ymin, th + 5)
        cv2.rectangle(frame, (xmin, y_text - th - 2), (xmin + tw + 2, y_text + 2), (0, 255, 0), -1)
        cv2.putText(frame, caption, (xmin + 1, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw mAP text (proxy if GT not available)
    map_text = 'mAP: N/A' if map_value is None else f'mAP: {map_value:.3f}'
    cv2.putText(frame, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 255), 2)

    return frame


def compute_map_proxy(class_conf_scores):
    # IMPORTANT: Real mAP requires ground-truth annotations to compute precision-recall.
    # This function computes a proxy by averaging detection confidences per class,
    # then taking the mean across classes observed so far.
    # It is only a placeholder when GT is unavailable.
    if not class_conf_scores:
        return None
    ap_values = []
    for scores in class_conf_scores.values():
        if len(scores) > 0:
            ap_values.append(float(np.mean(scores)))
    if len(ap_values) == 0:
        return None
    return float(np.mean(ap_values))


def main():
    # Step 1: Setup
    labels = load_labels(label_path)
    interpreter = make_interpreter(model_path)
    input_details = interpreter.get_input_details()[0]

    # Video I/O
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'ERROR: Unable to open input video: {input_path}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f'ERROR: Unable to open output video for write: {output_path}')
        cap.release()
        return

    # Stats
    class_conf_scores = {}  # {class_id: [scores]}
    frame_count = 0
    total_infer_time = 0.0

    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Step 2: Preprocessing
        input_data = preprocess_frame(frame, input_details)

        # Step 3: Inference
        set_input_tensor(interpreter, input_data)
        t0 = time.time()
        interpreter.invoke()
        infer_time = (time.time() - t0)
        total_infer_time += infer_time

        # Extract outputs
        boxes, classes, scores, num = get_output_tensors(interpreter)

        # Step 4: Output handling - decode detections
        detections = []
        det_count = int(np.squeeze(num).astype(np.int32)) if num is not None else scores.shape[1]
        for i in range(det_count):
            score = float(scores[0, i]) if scores is not None else 0.0
            if score < confidence_threshold:
                continue
            cls_id = int(classes[0, i]) if classes is not None else -1
            box = boxes[0, i] if boxes is not None else [0, 0, 0, 0]

            # Convert normalized box to pixel coords
            ymin = max(0, min(height - 1, int(box[0] * height)))
            xmin = max(0, min(width - 1, int(box[1] * width)))
            ymax = max(0, min(height - 1, int(box[2] * height)))
            xmax = max(0, min(width - 1, int(box[3] * width)))

            # Ensure box has area > 0
            if xmax <= xmin or ymax <= ymin:
                continue

            detections.append({
                'box': (ymin, xmin, ymax, xmax),
                'class_id': cls_id,
                'score': score
            })

            # Accumulate scores for proxy mAP computation
            if cls_id not in class_conf_scores:
                class_conf_scores[cls_id] = []
            class_conf_scores[cls_id].append(score)

        # Compute mAP proxy (see note in compute_map_proxy)
        map_proxy = compute_map_proxy(class_conf_scores)

        # Draw and write frame
        annotated = draw_detections(frame.copy(), detections, labels, map_proxy)
        writer.write(annotated)

    # Cleanup
    cap.release()
    writer.release()

    # Summary
    avg_infer_ms = (total_infer_time / frame_count * 1000.0) if frame_count > 0 else 0.0
    final_map_proxy = compute_map_proxy(class_conf_scores)
    print('Processing complete.')
    print(f'Frames processed: {frame_count}')
    print(f'Average inference time: {avg_infer_ms:.2f} ms')
    if final_map_proxy is None:
        print('mAP: N/A (ground truth not provided; proxy metric could not be computed)')
    else:
        print(f'mAP (proxy): {final_map_proxy:.3f}')
    print(f'Output saved to: {output_path}')


if __name__ == "__main__":
    main()