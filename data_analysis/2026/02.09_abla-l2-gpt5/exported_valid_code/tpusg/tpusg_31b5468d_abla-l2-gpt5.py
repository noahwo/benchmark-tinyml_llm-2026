import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# --------------------------
# Utilities
# --------------------------
def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try "index label" format, else fallback to line order
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[i] = line
    return labels

def make_interpreter(model_file):
    return Interpreter(
        model_path=model_file,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )

def preprocess_frame(frame_bgr, input_size, input_dtype):
    h, w = input_size
    # Convert BGR to RGB and resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))
    input_data = np.expand_dims(resized, axis=0)
    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)
    return input_data

def run_inference(interpreter, input_data, input_index):
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    # Typical EdgeTPU detection model outputs:
    # 0: boxes [1, num, 4], 1: classes [1, num], 2: scores [1, num], 3: num [1]
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores).astype(np.float32)
    num = int(np.squeeze(num))
    return boxes, classes, scores, num

def class_color(class_id):
    # Deterministic color per class id
    rng = np.random.RandomState(class_id)
    color = rng.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def draw_detections(frame, detections, labels, map_value=None):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        score = det['score']
        color = class_color(class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = labels.get(class_id, str(class_id))
        text = "{}: {:.2f}".format(label_text, score)
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if map_value is not None:
        map_text = "mAP: {:.2f}%".format(map_value * 100.0)
        cv2.putText(frame, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 2, cv2.LINE_AA)

def compute_ap_11pt(precisions, recalls):
    # VOC 2007 11-point interpolation
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:  # 0.0, 0.1, ..., 1.0
        p = 0.0
        for pr, rc in zip(precisions, recalls):
            if rc >= t and pr > p:
                p = pr
        ap += p
    return ap / 11.0

def compute_map_pseudo(dets_by_class):
    # Pseudo mAP without ground truth:
    # Assume at most 1 true object per frame per class.
    # For each class, within each frame: highest-score detection -> TP, others -> FP.
    aps = []
    for class_id, frame_scores in dets_by_class.items():
        # frame_scores: dict frame_id -> list of scores
        total_positives = 0
        scored_tuples = []  # (score, is_tp)
        for _, scores in frame_scores.items():
            if not scores:
                continue
            scores_sorted = sorted(scores, reverse=True)
            # one TP per frame (the top), rest FP
            scored_tuples.append((scores_sorted[0], 1))
            for s in scores_sorted[1:]:
                scored_tuples.append((s, 0))
            total_positives += 1
        if total_positives == 0 or len(scored_tuples) == 0:
            continue
        # Sort all detections by score descending
        scored_tuples.sort(key=lambda x: x[0], reverse=True)
        precisions = []
        recalls = []
        tp_cum = 0
        fp_cum = 0
        for s, is_tp in scored_tuples:
            if is_tp == 1:
                tp_cum += 1
            else:
                fp_cum += 1
            prec = tp_cum / float(tp_cum + fp_cum)
            rec = tp_cum / float(total_positives)
            precisions.append(prec)
            recalls.append(rec)
        ap = compute_ap_11pt(precisions, recalls)
        aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# --------------------------
# Main application
# --------------------------
def main():
    # Setup: load interpreter and labels, open input
    labels = load_labels(label_path)
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    in_h = int(input_details['shape'][1])
    in_w = int(input_details['shape'][2])
    in_dtype = input_details['dtype']
    in_index = input_details['index']

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video: {}".format(input_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # First pass: run inference, collect detections per frame and per class for mAP
    frame_detections = []  # list index by frame_id: list of detection dicts
    dets_by_class = {}     # class_id -> {frame_id: [scores]}
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_frame(frame, (in_h, in_w), in_dtype)
        boxes, classes, scores, num = run_inference(interpreter, input_data, in_index)

        detections_this_frame = []
        for i in range(num):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            class_id = int(classes[i])
            # Boxes are normalized [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = max(0, min(src_w - 1, int(xmin * src_w)))
            y1 = max(0, min(src_h - 1, int(ymin * src_h)))
            x2 = max(0, min(src_w - 1, int(xmax * src_w)))
            y2 = max(0, min(src_h - 1, int(ymax * src_h)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections_this_frame.append({
                'bbox': (x1, y1, x2, y2),
                'class_id': class_id,
                'score': score
            })
            # For mAP (pseudo) accumulation
            if class_id not in dets_by_class:
                dets_by_class[class_id] = {}
            if frame_id not in dets_by_class[class_id]:
                dets_by_class[class_id][frame_id] = []
            dets_by_class[class_id][frame_id].append(score)

        frame_detections.append(detections_this_frame)
        frame_id += 1

    cap.release()

    # Compute mAP (pseudo, due to lack of ground truth)
    map_value = compute_map_pseudo(dets_by_class)

    # Second pass: write output video with rectangles, labels, and mAP overlay
    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (src_w, src_h))
    cap2 = cv2.VideoCapture(input_path)
    if not cap2.isOpened():
        writer.release()
        raise RuntimeError("Cannot reopen input video for writing: {}".format(input_path))

    frame_id = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        dets = frame_detections[frame_id] if frame_id < len(frame_detections) else []
        draw_detections(frame, dets, labels, map_value=map_value)
        writer.write(frame)
        frame_id += 1

    cap2.release()
    writer.release()

if __name__ == "__main__":
    main()