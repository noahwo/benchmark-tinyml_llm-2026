import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            labels.append(name)
    return labels

def setup_interpreter(model_path, num_threads=4):
    # Initialize the TFLite interpreter
    try:
        interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    except TypeError:
        # Fallback if num_threads is not supported
        interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_output_indices(interpreter):
    # Identify output indices by shape heuristics (SSD models typically have boxes, classes, scores, num)
    output_details = interpreter.get_output_details()
    idx = {'boxes': None, 'classes': None, 'scores': None, 'num': None}
    for od in output_details:
        shape = od['shape']
        if len(shape) == 2 and shape[-1] >= 1:
            # Some variants may present (N,4) for boxes or (N,) for classes/scores; handle robustly below
            pass
        if len(shape) == 3 and shape[-1] == 4:
            idx['boxes'] = od['index']
        elif len(shape) == 2 and shape[-1] > 4:
            # classes or scores; will differentiate by dtype (classes are often float with integers)
            # Can't rely purely on dtype. Will assign after reading to determine meaning by value range if needed.
            pass
        elif len(shape) == 2 and shape[-1] >= 1:
            pass
        elif len(shape) == 1 and shape[0] == 1:
            idx['num'] = od['index']

    # If not resolved by above, map by name hints if present
    if any(v is None for v in idx.values()):
        for od in output_details:
            name = od.get('name', '').lower()
            if idx['boxes'] is None and 'boxes' in name:
                idx['boxes'] = od['index']
            elif idx['classes'] is None and 'classes' in name:
                idx['classes'] = od['index']
            elif idx['scores'] is None and ('scores' in name or 'scores' in name):
                idx['scores'] = od['index']
            elif idx['num'] is None and ('num' in name or 'detections' in name):
                # many models use 'num_detections'
                if od['shape'] == (1,) or od['shape'] == [1]:
                    idx['num'] = od['index']

    # Final fallback: assign remaining by shapes typical to SSD
    output_details = interpreter.get_output_details()
    # collect candidates
    candidates = {od['index']: od['shape'] for od in output_details}
    # Boxes
    if idx['boxes'] is None:
        for i, shp in candidates.items():
            if len(shp) == 3 and shp[-1] == 4:
                idx['boxes'] = i
                break
    # num
    if idx['num'] is None:
        for i, shp in candidates.items():
            if (len(shp) == 1 and shp[0] == 1) or (len(shp) == 2 and shp == [1, 1]):
                idx['num'] = i
                break
    # The remaining two outputs should be classes and scores, both shaped (1, N)
    remaining = [i for i in candidates.keys() if i not in (idx['boxes'], idx['num'])]
    # Try to distinguish by dtype or value inspection after inference; for now, just return indices and resolve later
    # We'll resolve after first inference if needed.
    return idx, remaining

def preprocess_frame(frame, input_details):
    # Convert BGR to RGB and resize to model input size with no letterbox
    _, in_h, in_w, _ = input_details[0]['shape']
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        # Assume uint8 quantized input
        input_data = resized.astype(np.uint8)

    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def draw_detections(frame, boxes, classes, scores, num, labels, label_offset, threshold):
    h, w = frame.shape[:2]
    num = int(num)
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        box = boxes[i]  # [ymin, xmin, ymax, xmax] normalized
        ymin = max(0, int(box[0] * h))
        xmin = max(0, int(box[1] * w))
        ymax = min(h - 1, int(box[2] * h))
        xmax = min(w - 1, int(box[3] * w))

        cls_id = int(classes[i])
        label_idx = cls_id + label_offset
        if 0 <= label_idx < len(labels):
            label = labels[label_idx]
        else:
            label = f"id {cls_id}"

        color = (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        caption = f"{label}: {score*100:.1f}%"
        ((text_w, text_h), _) = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_bg_tl = (xmin, max(0, ymin - text_h - 6))
        text_bg_br = (xmin + text_w + 6, ymin)
        cv2.rectangle(frame, text_bg_tl, text_bg_br, color, thickness=-1)
        cv2.putText(frame, caption, (xmin + 3, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    # 1. setup
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(LABEL_PATH):
        raise SystemExit(f"Label file not found: {LABEL_PATH}")
    if not os.path.isfile(INPUT_PATH):
        raise SystemExit(f"Input video not found: {INPUT_PATH}")

    labels = load_labels(LABEL_PATH)
    label_offset = 1 if len(labels) > 0 and labels[0].strip().lower() in ("???", "background") else 0

    interpreter = setup_interpreter(MODEL_PATH, num_threads=4)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    out_map, unresolved = get_output_indices(interpreter)

    # Video IO setup
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open output video for writing: {OUTPUT_PATH}")

    frame_count = 0
    t_start = time.perf_counter()
    inference_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 2. preprocessing
            input_data = preprocess_frame(frame, input_details)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # 3. inference
            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()
            inference_times.append(t1 - t0)

            # Retrieve outputs
            # First, get the boxes and num if mapped
            boxes = interpreter.get_tensor(out_map['boxes']) if out_map['boxes'] is not None else None
            num_dets = interpreter.get_tensor(out_map['num']) if out_map['num'] is not None else None

            # Resolve remaining outputs (classes and scores)
            classes = None
            scores = None
            if unresolved:
                tensors = [interpreter.get_tensor(i) for i in unresolved]
                # Expect shapes [1, N] for both; identify by value ranges
                for arr in tensors:
                    arr1 = np.squeeze(arr)
                    # Heuristic: classes are near small integers, scores are floats in [0,1]
                    if arr1.dtype in (np.float32, np.float64):
                        if np.all((arr1 >= 0.0) & (arr1 <= 1.0)):
                            # could be scores
                            if scores is None:
                                scores = arr1
                            else:
                                # if collision, choose the one with mean closer to high confidences
                                if np.mean(arr1) > np.mean(scores):
                                    scores = arr1
                        else:
                            # likely classes encoded as floats
                            if classes is None:
                                classes = arr1.astype(np.int32)
                            else:
                                # choose the one with larger unique integer count as classes
                                if len(np.unique(arr1.astype(np.int32))) > len(np.unique(classes)):
                                    classes = arr1.astype(np.int32)
                    else:
                        # integer types likely classes
                        classes = arr1.astype(np.int32)
            # If mapping known names or singletons exist in out_map, fetch directly
            if classes is None:
                # try by name lookup
                for od in output_details:
                    name = od.get('name', '').lower()
                    if 'class' in name:
                        classes = np.squeeze(interpreter.get_tensor(od['index'])).astype(np.int32)
                        break
            if scores is None:
                for od in output_details:
                    name = od.get('name', '').lower()
                    if 'score' in name:
                        scores = np.squeeze(interpreter.get_tensor(od['index'])).astype(np.float32)
                        break
            if boxes is None or num_dets is None or classes is None or scores is None:
                # Fallback to common order: [boxes, classes, scores, num_detections]
                out = [interpreter.get_tensor(od['index']) for od in output_details]
                # Attempt standard unpack
                if len(out) >= 4:
                    boxes = out[0]
                    classes = np.squeeze(out[1]).astype(np.int32)
                    scores = np.squeeze(out[2]).astype(np.float32)
                    num_dets = out[3]
                else:
                    raise SystemExit("Unable to resolve model outputs for SSD detection.")

            boxes = np.squeeze(boxes)
            if boxes.ndim == 1 and boxes.size == 4:
                boxes = np.expand_dims(boxes, axis=0)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores).astype(np.float32)
            num = int(np.squeeze(num_dets))

            # 4. output handling (draw detections and write video)
            draw_detections(frame, boxes, classes, scores, num, labels, label_offset, CONFIDENCE_THRESHOLD)
            writer.write(frame)

            # Optional: simple progress output every 30 frames
            if frame_count % 30 == 0:
                avg_inf = (sum(inference_times[-30:]) / min(30, len(inference_times))) if inference_times else 0.0
                print(f"Processed {frame_count} frames | Avg inference: {avg_inf*1000:.1f} ms")

    finally:
        cap.release()
        writer.release()

    t_end = time.perf_counter()
    total_time = t_end - t_start
    avg_inf_ms = (sum(inference_times) / len(inference_times) * 1000.0) if inference_times else 0.0
    print(f"Done. Frames: {frame_count}, Total time: {total_time:.2f}s, Avg inference: {avg_inf_ms:.1f} ms/frame")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()