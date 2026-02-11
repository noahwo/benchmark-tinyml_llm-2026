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
EDGETPU_DELEGATE_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(label_path):
    labels = {}
    if not os.path.isfile(label_path):
        return labels
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        # Build id: label mapping. Assume each line is a label in order.
        for idx, line in enumerate(lines):
            if not line:
                continue
            labels[idx] = line
    return labels


def make_color_for_id(class_id):
    # Deterministic pseudo-color for a class id
    # Simple hashing into BGR space
    b = (37 * (class_id + 1)) % 255
    g = (17 * (class_id + 7)) % 255
    r = (29 * (class_id + 13)) % 255
    return int(b), int(g), int(r)


def prepare_interpreter(model_path, delegate_path):
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(delegate_path)]
    )
    interpreter.allocate_tensors()
    return interpreter


def get_input_info(interpreter):
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    ih, iw, ic = input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]
    input_dtype = input_details[0]['dtype']
    return input_index, (iw, ih, ic), input_dtype


def find_output_indices(interpreter):
    # Try to find output tensors by common names, otherwise fall back to ordering.
    output_details = interpreter.get_output_details()
    idx_boxes = idx_scores = idx_classes = idx_num = None
    for od in output_details:
        name = od.get('name', '').lower()
        if 'box' in name and idx_boxes is None:
            idx_boxes = od['index']
        elif 'score' in name and idx_scores is None:
            idx_scores = od['index']
        elif 'class' in name and idx_classes is None:
            idx_classes = od['index']
        elif 'num' in name and idx_num is None:
            idx_num = od['index']
    # Fallback: assume typical order [boxes, classes, scores, num] or variants
    if idx_boxes is None or idx_scores is None or idx_classes is None or idx_num is None:
        # Heuristic: assign based on tensor shapes
        # boxes: 3D [1, N, 4]
        # classes: 2D [1, N]
        # scores: 2D [1, N]
        # num: 1D [1]
        for od in output_details:
            shape = od.get('shape', [])
            name = od.get('name', '').lower()
            if len(shape) == 3 and shape[-1] == 4:
                idx_boxes = od['index']
        twos = [od for od in output_details if len(od.get('shape', [])) == 2]
        ones = [od for od in output_details if len(od.get('shape', [])) == 1]
        if twos:
            # Distinguish classes vs scores by dtype (classes often float then castable)
            # If both are float32, use name hints
            for od in twos:
                n = od.get('name', '').lower()
                if 'class' in n and idx_classes is None:
                    idx_classes = od['index']
                elif 'score' in n and idx_scores is None:
                    idx_scores = od['index']
            # If still missing, assign from remaining
            remaining = [od for od in twos if od['index'] not in (idx_classes if idx_classes is not None else -1, idx_scores if idx_scores is not None else -1)]
            for od in remaining:
                if idx_scores is None:
                    idx_scores = od['index']
                elif idx_classes is None:
                    idx_classes = od['index']
        if ones:
            idx_num = ones[0]['index']
    return idx_boxes, idx_scores, idx_classes, idx_num


def preprocess_frame(frame_bgr, input_size, input_dtype):
    iw, ih, _ = input_size
    # Resize and convert to RGB
    resized = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        input_tensor = (rgb.astype(np.float32) / 255.0)
    else:
        input_tensor = rgb.astype(input_dtype)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def run_inference(interpreter, input_index, input_tensor, idx_boxes, idx_scores, idx_classes, idx_num):
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    # Retrieve outputs
    boxes = interpreter.get_tensor(idx_boxes) if idx_boxes is not None else None
    scores = interpreter.get_tensor(idx_scores) if idx_scores is not None else None
    classes = interpreter.get_tensor(idx_classes) if idx_classes is not None else None
    num = interpreter.get_tensor(idx_num) if idx_num is not None else None

    # Squeeze batch dimension
    if boxes is not None and boxes.ndim >= 3:
        boxes = boxes[0]
    if scores is not None and scores.ndim >= 2:
        scores = scores[0]
    if classes is not None and classes.ndim >= 2:
        classes = classes[0]
    if num is not None:
        try:
            num = int(num[0])
        except Exception:
            num = None

    # If num not available, infer from scores length
    if num is None and scores is not None:
        num = scores.shape[0]

    # Cast types
    if classes is not None:
        classes = classes.astype(np.int32, copy=False)
    if scores is None or boxes is None or classes is None or num is None:
        return [], [], [], 0

    return boxes, scores, classes, num


def draw_detections(frame, detections, labels, running_map):
    h, w = frame.shape[:2]
    for det in detections:
        (ymin, xmin, ymax, xmax), cls_id, score = det
        # Convert normalized to absolute coordinates
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        color = make_color_for_id(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = labels.get(cls_id, f"id:{cls_id}")
        text = f"{label_text}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline)
        cv2.rectangle(frame, (x1, y_text), (x1 + tw + 4, y_text + th + baseline), color, thickness=-1)
        cv2.putText(frame, text, (x1 + 2, y_text + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Overlay running "mAP" (proxy due to lack of GT) at top-left
    map_text = f"mAP: {running_map:.3f}" if running_map is not None else "mAP: N/A"
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2, lineType=cv2.LINE_AA)


def main():
    # Load labels
    labels = load_labels(LABEL_PATH)

    # Prepare TFLite interpreter with EdgeTPU
    interpreter = prepare_interpreter(MODEL_PATH, EDGETPU_DELEGATE_PATH)
    input_index, input_size, input_dtype = get_input_info(interpreter)
    idx_boxes, idx_scores, idx_classes, idx_num = find_output_indices(interpreter)

    # Prepare video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Unable to open input video: {INPUT_PATH}")
        return

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (in_w, in_h))
    if not writer.isOpened():
        print(f"ERROR: Unable to open output video for write: {OUTPUT_PATH}")
        cap.release()
        return

    # Metrics accumulators (proxy mAP: mean of detection confidences above threshold)
    total_conf_sum = 0.0
    total_det_count = 0

    frame_index = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Preprocess
        input_tensor = preprocess_frame(frame, input_size, input_dtype)

        # Inference
        boxes, scores, classes, num = run_inference(interpreter, input_index, input_tensor, idx_boxes, idx_scores, idx_classes, idx_num)

        # Collect detections above threshold
        detections = []
        for i in range(num):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            box = boxes[i]  # [ymin, xmin, ymax, xmax] normalized
            cls_id = int(classes[i])
            detections.append((box, cls_id, score))

            # Update proxy mAP stats
            total_conf_sum += score
            total_det_count += 1

        running_map = (total_conf_sum / total_det_count) if total_det_count > 0 else None

        # Draw
        draw_detections(frame, detections, labels, running_map)

        # Optional overlay of frame counter
        cv2.putText(frame, f"Frame: {frame_index}", (10, in_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, lineType=cv2.LINE_AA)

        # Write frame
        writer.write(frame)

    elapsed = time.time() - t0
    proc_fps = (frame_index / elapsed) if elapsed > 0 else 0.0

    # Cleanup
    cap.release()
    writer.release()

    # Print summary
    if total_det_count > 0:
        map_value = total_conf_sum / total_det_count
        print(f"Processed {frame_index} frames in {elapsed:.2f}s ({proc_fps:.2f} FPS).")
        print(f"Detections above threshold: {total_det_count}")
        print(f"Computed mAP (proxy from confidences): {map_value:.4f}")
    else:
        print(f"Processed {frame_index} frames in {elapsed:.2f}s ({proc_fps:.2f} FPS).")
        print("No detections above threshold; mAP not available.")

    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()