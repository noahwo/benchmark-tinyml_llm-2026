import os
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def prepare_input(image_bgr, input_shape, input_dtype, quant_params):
    # input_shape: [1, h, w, c]
    _, ih, iw, _ = input_shape
    # Convert to RGB and resize to model input
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    input_tensor = resized.astype(np.float32)
    if input_dtype == np.float32:
        # Normalize to [0,1]
        input_tensor = input_tensor / 255.0
    else:
        # Quantized input (uint8) with scale/zero_point
        scale, zero_point = quant_params if quant_params is not None else (1.0, 0)
        if scale and scale > 0:
            input_tensor = np.round(input_tensor / scale + zero_point).astype(np.uint8)
        else:
            input_tensor = input_tensor.astype(np.uint8)
    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def extract_output(interpreter):
    # Attempt to robustly fetch boxes, classes, scores, num
    output_details = interpreter.get_output_details()
    tensors = {}
    # Retrieve all tensors first
    for od in output_details:
        name = od.get('name', '').lower()
        data = interpreter.get_tensor(od['index'])
        tensors[name] = data

    # Heuristic mapping by names
    boxes = None
    classes = None
    scores = None
    num = None

    def find_by_keyword(keyword):
        for name, data in tensors.items():
            if keyword in name:
                return data
        return None

    boxes = find_by_keyword('box')
    if boxes is None:
        boxes = find_by_keyword('bbox')
    classes = find_by_keyword('class')
    scores = find_by_keyword('score')
    num = find_by_keyword('num')

    # Fallback to positional assumptions
    if boxes is None or classes is None or scores is None:
        # Positional guess: [boxes, classes, scores, num]
        # Sort by rank/shape if names not informative
        ods = interpreter.get_output_details()
        arrs = [interpreter.get_tensor(od['index']) for od in ods]
        # Find boxes: tensor with shape [1, N, 4]
        for a in arrs:
            if isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[-1] == 4:
                boxes = a
                break
        # Find num: tensor with shape [1] or scalar-like
        for a in arrs:
            if isinstance(a, np.ndarray) and a.size == 1 and a.ndim in (0, 1):
                num = a
                break
        # Remaining two of shape [1, N] -> classes and scores (float32)
        candidates = [a for a in arrs if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[0] == 1]
        if len(candidates) >= 2:
            # Heuristic: scores are [0,1] floats
            cand0, cand1 = candidates[0], candidates[1]
            def looks_like_scores(x):
                return x.dtype.kind == 'f' and np.all(x <= 1.0001) and np.all(x >= -0.0001)
            if looks_like_scores(cand0) and not looks_like_scores(cand1):
                scores, classes = cand0, cand1
            elif looks_like_scores(cand1) and not looks_like_scores(cand0):
                scores, classes = cand1, cand0
            else:
                # Default: assume order scores then classes
                scores, classes = cand0, cand1

    # Squeeze leading batch if present
    if isinstance(boxes, np.ndarray) and boxes.ndim == 3:
        boxes = boxes[0]
    if isinstance(classes, np.ndarray) and classes.ndim == 2:
        classes = classes[0]
    if isinstance(scores, np.ndarray) and scores.ndim == 2:
        scores = scores[0]
    if isinstance(num, np.ndarray):
        num = int(num.flatten()[0])

    # Final safeguards
    if boxes is None or classes is None or scores is None or num is None:
        raise RuntimeError("Failed to parse model outputs (boxes/classes/scores/num).")

    # Classes may come as float; cast to int for indexing
    classes = classes.astype(np.int32, copy=False)
    return boxes, classes, scores, num

def compute_map_proxy(conf_by_class):
    # Proxy mAP: average of mean confidences per class that had at least one detection
    # Note: This is a placeholder proxy since no ground-truth annotations are provided.
    per_class_avgs = []
    for _, confs in conf_by_class.items():
        if len(confs) > 0:
            per_class_avgs.append(float(np.mean(confs)))
    if len(per_class_avgs) == 0:
        return 0.0
    return float(np.mean(per_class_avgs))

def get_color_for_class(class_id):
    # Deterministic color per class id
    rng = np.random.RandomState(class_id * 9973 + 12345)
    color = tuple(int(x) for x in rng.randint(0, 255, size=3))
    return color

def draw_labelled_box(frame, box, label_text, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # Text background
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    th_full = th + baseline
    x2_bg = min(frame.shape[1] - 1, x1 + tw + 6)
    y2_bg = max(0, y1 - th_full - 4)
    cv2.rectangle(frame, (x1, y1 - th_full - 6), (x2_bg, y1), color, -1)
    cv2.putText(frame, label_text, (x1 + 3, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load labels
    labels = load_labels(label_path)
    label_offset = 1 if len(labels) > 0 and labels[0].strip().lower() == '???' else 0

    # Initialize TFLite Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # [1, h, w, c]
    input_dtype = input_details[0]['dtype']
    quant_params = input_details[0].get('quantization', None)
    if isinstance(quant_params, (list, tuple)) and len(quant_params) == 2:
        pass
    else:
        quant_params = None

    # Video IO setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 0:
        in_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, in_fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # Statistics
    conf_by_class = {}  # class_id -> list of confidences (above threshold)
    infer_times = []
    total_frames = 0

    # Processing loop
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        total_frames += 1

        # Prepare input
        input_tensor = prepare_input(frame_bgr, input_shape, input_dtype, quant_params)

        # Set input and run inference
        interpreter.set_tensor(input_index, input_tensor)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        infer_times.append(t1 - t0)

        # Extract and postprocess outputs
        try:
            boxes, classes, scores, num = extract_output(interpreter)
        except Exception as e:
            # If output parsing fails, write frame untouched but continue
            cv2.putText(frame_bgr, f"Inference output error: {e}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            out.write(frame_bgr)
            continue

        # Draw detections
        for i in range(min(num, boxes.shape[0], scores.shape[0], classes.shape[0])):
            score = float(scores[i])
            if score < confidence_threshold:
                continue

            cls_id_raw = int(classes[i])
            cls_id = cls_id_raw + (label_offset)
            # Map cls_id to label index space
            label_idx = cls_id
            if label_idx < 0 or label_idx >= len(labels):
                label_text = f"id:{cls_id_raw} {score:.2f}"
            else:
                label_text = f"{labels[label_idx]} {score:.2f}"

            # Box coordinates are normalized ymin, xmin, ymax, xmax
            y_min, x_min, y_max, x_max = boxes[i]
            x1 = max(0, min(width - 1, int(x_min * width)))
            y1 = max(0, min(height - 1, int(y_min * height)))
            x2 = max(0, min(width - 1, int(x_max * width)))
            y2 = max(0, min(height - 1, int(y_max * height)))
            # Ensure proper ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            color = get_color_for_class(cls_id_raw)
            draw_labelled_box(frame_bgr, (x1, y1, x2, y2), label_text, color, thickness=2)

            # Update proxy mAP stats
            if cls_id_raw not in conf_by_class:
                conf_by_class[cls_id_raw] = []
            conf_by_class[cls_id_raw].append(score)

        # Compute and overlay proxy mAP and FPS
        map_proxy = compute_map_proxy(conf_by_class)
        avg_infer = np.mean(infer_times) if len(infer_times) > 0 else 0.0
        fps_est = 1.0 / avg_infer if avg_infer > 0 else 0.0

        overlay_text_1 = f"Frames: {total_frames}  mAP(proxy): {map_proxy:.3f}"
        overlay_text_2 = f"FPS(est): {fps_est:.1f}  Threshold: {confidence_threshold:.2f}"
        cv2.putText(frame_bgr, overlay_text_1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, overlay_text_2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)

        out.write(frame_bgr)

    # Release resources
    cap.release()
    out.release()

    # Final stats
    final_map_proxy = compute_map_proxy(conf_by_class)
    avg_infer = float(np.mean(infer_times)) if len(infer_times) > 0 else 0.0
    print("Processing complete.")
    print(f"Total frames: {total_frames}")
    print(f"Average inference time per frame: {avg_infer*1000:.2f} ms")
    print(f"Estimated FPS: {1.0/avg_infer if avg_infer>0 else 0.0:.2f}")
    print(f"Proxy mAP (no ground-truth available): {final_map_proxy:.3f}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()