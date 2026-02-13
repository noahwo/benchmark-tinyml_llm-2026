import os
import time
import numpy as np
import cv2

# TFLite runtime for EdgeTPU
from tflite_runtime.interpreter import Interpreter, load_delegate

# ==========================
# Configuration Parameters
# ==========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

INPUT_DESCRIPTION = "Read a single video file from the given input_path"
OUTPUT_DESCRIPTION = "Output the video file with rectangles drew on the detected objects, along with texts of labels and calculated mAP(mean average precision)"

EDGETPU_SHARED_LIB = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"

# ==========================
# Utilities
# ==========================
def load_labels(label_path):
    labels = {}
    try:
        with open(label_path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                labels[idx] = line
    except Exception:
        # If label file cannot be read, fallback to empty labels
        labels = {}
    return labels

def make_interpreter(model_path, edgetpu_lib):
    delegate = load_delegate(edgetpu_lib)
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    return interpreter

def get_input_size(interpreter):
    input_details = interpreter.get_input_details()[0]
    _, height, width, channels = input_details['shape']
    return width, height, channels, input_details['dtype']

def set_input_tensor(interpreter, data):
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, data)

def preprocess_frame(frame, in_w, in_h, in_dtype):
    # Convert BGR to RGB and resize to model input size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    if in_dtype == np.uint8:
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
    else:
        # Fallback for float models (scale to [0,1])
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return input_data

def _dequantize(arr, detail):
    # Dequantize if needed
    if 'quantization' in detail:
        scale, zero_point = detail['quantization']
        if scale and scale > 0:
            return scale * (arr.astype(np.float32) - zero_point)
    return arr

def extract_detections(interpreter, frame_w, frame_h):
    """
    Returns list of detections: [{'box':(x1,y1,x2,y2), 'score':float, 'class_id':int}]
    """
    output_details = interpreter.get_output_details()
    outputs = [interpreter.get_tensor(d['index']) for d in output_details]

    # Try to identify outputs by name
    boxes = classes = scores = num = None
    for d, arr in zip(output_details, outputs):
        name = d.get('name', '')
        if isinstance(name, bytes):
            name = name.decode('ascii', errors='ignore')
        arr = _dequantize(arr, d)
        if 'box' in name or 'boxes' in name:
            boxes = arr[0]
        elif 'score' in name or 'scores' in name:
            scores = arr[0]
        elif 'class' in name or 'classes' in name:
            classes = arr[0]
        elif 'num' in name:
            try:
                num = int(np.squeeze(arr))
            except Exception:
                num = None

    # Fallback heuristic if names were not helpful
    if boxes is None or classes is None or scores is None:
        # Identify boxes by shape (1, N, 4)
        for d, arr in zip(output_details, outputs):
            arr = _dequantize(arr, d)
            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 4:
                boxes = arr[0]
        # Identify scores and classes among (1, N) tensors
        one_n_tensors = []
        for d, arr in zip(output_details, outputs):
            arr = _dequantize(arr, d)
            if arr.ndim == 2 and arr.shape[0] == 1:
                one_n_tensors.append(arr[0])
        if len(one_n_tensors) >= 2:
            # Choose the array with values mostly in [0,1] as scores
            scores_candidate, classes_candidate = None, None
            if np.mean((one_n_tensors[0] >= 0) & (one_n_tensors[0] <= 1)) > 0.8:
                scores_candidate = one_n_tensors[0]
                classes_candidate = one_n_tensors[1]
            elif np.mean((one_n_tensors[1] >= 0) & (one_n_tensors[1] <= 1)) > 0.8:
                scores_candidate = one_n_tensors[1]
                classes_candidate = one_n_tensors[0]
            else:
                # Default order
                scores_candidate = one_n_tensors[0]
                classes_candidate = one_n_tensors[1]
            scores = scores_candidate
            classes = classes_candidate

    if boxes is None or classes is None or scores is None:
        return []

    if num is None:
        num = min(len(scores), len(classes), len(boxes))

    detections = []
    for i in range(int(num)):
        score = float(scores[i])
        cls = int(classes[i])
        y_min, x_min, y_max, x_max = boxes[i]
        # Convert normalized coords to absolute pixel coords
        x1 = int(max(0, min(1, x_min)) * frame_w)
        y1 = int(max(0, min(1, y_min)) * frame_h)
        x2 = int(max(0, min(1, x_max)) * frame_w)
        y2 = int(max(0, min(1, y_max)) * frame_h)
        # Ensure proper ordering
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({'box': (x1, y1, x2, y2), 'score': score, 'class_id': cls})
    return detections

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxA_area + boxB_area - inter_area)
    if denom <= 0:
        return 0.0
    return inter_area / denom

def nms_per_class(dets, iou_thresh=0.5):
    # Group by class
    dets_by_cls = {}
    for d in dets:
        dets_by_cls.setdefault(d['class_id'], []).append(d)

    kept = []
    for cls, items in dets_by_cls.items():
        # Sort by score descending
        items_sorted = sorted(items, key=lambda x: x['score'], reverse=True)
        selected = []
        while items_sorted:
            current = items_sorted.pop(0)
            selected.append(current)
            remaining = []
            for other in items_sorted:
                if iou(current['box'], other['box']) <= iou_thresh:
                    remaining.append(other)
            items_sorted = remaining
        kept.extend(selected)
    return kept

# ==========================
# Proxy mAP computation (temporal consistency-based)
# ==========================
class TemporalMAP:
    def __init__(self, thresholds=None, iou_threshold=0.5):
        # thresholds for confidence
        if thresholds is None:
            self.thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95
        else:
            self.thresholds = thresholds
        self.iou_threshold = iou_threshold
        # metrics[t][class_id] = {'TP': int, 'FP': int, 'FN': int}
        self.metrics = {t: {} for t in self.thresholds}
        self.prev_dets = []  # list of {'box','score','class_id'}

    def update(self, curr_dets):
        # Use NMS-normalized detections as input to metrics
        prev = self.prev_dets
        self.prev_dets = curr_dets

        if not prev:
            return  # no updates on first frame

        for t in self.thresholds:
            # Organize by class for both prev (as proxy GT) and current predictions
            prev_by_c = {}
            curr_by_c = {}
            for d in prev:
                if d['score'] >= t:
                    prev_by_c.setdefault(d['class_id'], []).append(d['box'])
            for d in curr_dets:
                if d['score'] >= t:
                    curr_by_c.setdefault(d['class_id'], []).append((d['box'], d['score']))

            all_classes = set(prev_by_c.keys()) | set(curr_by_c.keys())
            for cls in all_classes:
                gt_boxes = prev_by_c.get(cls, [])
                preds = curr_by_c.get(cls, [])
                preds = sorted(preds, key=lambda x: x[1], reverse=True)  # sort by score
                used_gt = set()
                TP = 0
                FP = 0
                for pred_box, _ in preds:
                    match_idx = -1
                    best_iou = 0.0
                    for gi, gt_box in enumerate(gt_boxes):
                        if gi in used_gt:
                            continue
                        i = iou(pred_box, gt_box)
                        if i > best_iou:
                            best_iou = i
                            match_idx = gi
                    if best_iou >= self.iou_threshold and match_idx >= 0:
                        TP += 1
                        used_gt.add(match_idx)
                    else:
                        FP += 1
                FN = len(gt_boxes) - len(used_gt)

                stats = self.metrics[t].setdefault(cls, {'TP': 0, 'FP': 0, 'FN': 0})
                stats['TP'] += TP
                stats['FP'] += FP
                stats['FN'] += FN

    def compute_map(self):
        # Compute AP per class as mean precision across thresholds; mAP is mean over classes
        class_aps = []
        for cls in self._all_classes_seen():
            precisions = []
            for t in self.thresholds:
                stats = self.metrics[t].get(cls, {'TP': 0, 'FP': 0, 'FN': 0})
                tp = stats['TP']
                fp = stats['FP']
                fn = stats['FN']
                denom = tp + fp
                if denom > 0:
                    precision = tp / float(denom)
                    precisions.append(precision)
            if precisions:
                ap = float(np.mean(precisions))
                class_aps.append(ap)
        if class_aps:
            return float(np.mean(class_aps))
        return 0.0

    def _all_classes_seen(self):
        classes = set()
        for t in self.thresholds:
            classes.update(self.metrics[t].keys())
        return classes

# ==========================
# Drawing
# ==========================
def draw_detections(frame, dets, labels, map_value):
    for d in dets:
        if d['score'] < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = d['box']
        cls = d['class_id']
        score = d['score']
        color = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = labels.get(cls, f"ID:{cls}")
        text = f"{label_text} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 2)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw mAP
    map_text = f"mAP: {map_value:.3f}"
    (tw, th), baseline = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (5, 5), (5 + tw + 10, 5 + th + baseline + 10), (50, 50, 50), -1)
    cv2.putText(frame, map_text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# ==========================
# Main
# ==========================
def main():
    print("Application: TFLite object detection with TPU")
    print("Deployment Target: Google Coral Dev Board")
    print(f"Input: {INPUT_PATH}")
    print(f" - {INPUT_DESCRIPTION}")
    print(f"Output: {OUTPUT_PATH}")
    print(f" - {OUTPUT_DESCRIPTION}")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # Prepare output directory
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU
    interpreter = make_interpreter(MODEL_PATH, EDGETPU_SHARED_LIB)
    in_w, in_h, in_ch, in_dtype = get_input_size(interpreter)

    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open input video: {INPUT_PATH}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # Metrics
    temporal_map = TemporalMAP()
    frame_count = 0
    t0 = time.time()
    last_map = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess
            input_data = preprocess_frame(frame, in_w, in_h, in_dtype)
            set_input_tensor(interpreter, input_data)

            # Inference
            start_inf = time.time()
            interpreter.invoke()
            inf_time = (time.time() - start_inf) * 1000.0  # ms

            # Postprocess: extract detections
            detections = extract_detections(interpreter, frame_w=out_w, frame_h=out_h)

            # NMS per class
            detections_nms = nms_per_class(detections, iou_thresh=0.5)

            # Update temporal mAP (proxy using previous frame as reference)
            temporal_map.update(detections_nms)
            last_map = temporal_map.compute_map()

            # Draw and write frame
            annotated = draw_detections(frame.copy(), detections_nms, labels, last_map)
            # Optionally draw inference time
            inf_text = f"Inference: {inf_time:.1f} ms"
            cv2.putText(annotated, inf_text, (10, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 255, 30), 1, cv2.LINE_AA)

            writer.write(annotated)

            # Optional: print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                fps_run = frame_count / max(1e-6, elapsed)
                print(f"[{frame_count} frames] {fps_run:.2f} FPS, current mAP: {last_map:.3f}")

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0
    overall_fps = frame_count / max(1e-6, total_time)
    final_map = temporal_map.compute_map()
    print(f"Processing complete. Frames: {frame_count}, Time: {total_time:.2f}s, Avg FPS: {overall_fps:.2f}")
    print(f"Final mAP: {final_map:.3f}")
    print(f"Saved annotated video to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()