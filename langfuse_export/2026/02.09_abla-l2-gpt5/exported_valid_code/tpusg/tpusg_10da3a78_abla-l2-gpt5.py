import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# Helper Functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        print("Warning: Label file not found at:", path)
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Try to parse formats:
    # "0 person", "0: person", "person"
    # Prefer explicit id if present
    for idx, line in enumerate(lines):
        if ':' in line:
            left, right = line.split(':', 1)
            left = left.strip()
            right = right.strip()
            if left.isdigit():
                labels[int(left)] = right
                continue
        # Space-delimited id and label
        parts = line.split()
        if len(parts) > 1 and parts[0].isdigit():
            lbl_id = int(parts[0])
            lbl = ' '.join(parts[1:])
            labels[lbl_id] = lbl
        else:
            # Fallback to sequential indexing
            labels[idx] = line
    return labels

def make_interpreter(model_file, edgetpu_lib="/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"):
    try:
        interpreter = Interpreter(
            model_path=model_file,
            experimental_delegates=[load_delegate(edgetpu_lib)]
        )
        return interpreter
    except Exception as e:
        raise RuntimeError(f"Failed to create TFLite interpreter with EdgeTPU delegate: {e}")

def get_output_indices(interpreter):
    # Attempt to robustly identify output tensors: boxes, classes, scores, count
    details = interpreter.get_output_details()
    boxes_idx = classes_idx = scores_idx = count_idx = None
    for i, d in enumerate(details):
        name = d.get('name', '')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='ignore')
        name_l = name.lower()
        shape = d.get('shape', [])
        if 'boxes' in name_l or (len(shape) == 3 and shape[-1] == 4):
            boxes_idx = i
        elif 'scores' in name_l:
            scores_idx = i
        elif 'classes' in name_l:
            classes_idx = i
        elif 'count' in name_l or 'num' in name_l:
            count_idx = i
    # Fallback to common SSD order if any missing
    if any(v is None for v in (boxes_idx, classes_idx, scores_idx, count_idx)):
        if len(details) >= 4:
            boxes_idx = 0 if boxes_idx is None else boxes_idx
            classes_idx = 1 if classes_idx is None else classes_idx
            scores_idx = 2 if scores_idx is None else scores_idx
            count_idx = 3 if count_idx is None else count_idx
        else:
            raise RuntimeError("Unable to determine output tensor indices for the detection model.")
    return boxes_idx, classes_idx, scores_idx, count_idx

def preprocess_frame_bgr_to_input(frame_bgr, input_w, input_h, input_dtype):
    # Convert BGR to RGB, resize to model input size, and convert dtype as needed
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h))
    if input_dtype == np.float32:
        inp = resized.astype(np.float32) / 255.0
    else:
        inp = resized.astype(np.uint8)
    return np.expand_dims(inp, axis=0)

def clip_bbox(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(0, min(xmax, w - 1))
    ymax = max(0, min(ymax, h - 1))
    return int(xmin), int(ymin), int(xmax), int(ymax)

def draw_detections(frame, detections, labels):
    for det in detections:
        xmin, ymin, xmax, ymax = det['bbox']
        score = det['score']
        cls = det['class_id']
        lbl = labels.get(cls, str(cls))
        color = (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        caption = f"{lbl}: {score:.2f}"
        # Background for text for readability
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, ymin - th - 4), (xmin + tw + 2, ymin), color, -1)
        cv2.putText(frame, caption, (xmin + 1, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def iou(boxA, boxB):
    # Boxes as [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    inter = inter_w * inter_h
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

def compute_map(detections, ground_truth, iou_threshold=0.5):
    # detections: list of dicts {'frame': int, 'class_id': int, 'score': float, 'bbox': [xmin,ymin,xmax,ymax]}
    # ground_truth: dict frame -> list of dicts {'class_id': int, 'bbox': [xmin,ymin,xmax,ymax]}
    # Returns mAP over classes present in ground truth. If none, returns None.
    # Organize ground truth by class and frame
    gt_by_class_frame = {}
    total_gts_by_class = {}
    frames_with_gt = set()
    for fidx, gts in ground_truth.items():
        for gt in gts:
            c = gt['class_id']
            gt_by_class_frame.setdefault(c, {}).setdefault(fidx, []).append({'bbox': gt['bbox'], 'matched': False})
            total_gts_by_class[c] = total_gts_by_class.get(c, 0) + 1
            frames_with_gt.add(fidx)

    if not total_gts_by_class:
        return None

    # Organize detections by class
    dets_by_class = {}
    for d in detections:
        c = d['class_id']
        dets_by_class.setdefault(c, []).append(d)

    ap_list = []
    for c, n_gt in total_gts_by_class.items():
        if n_gt == 0:
            continue
        # Gather detections of this class
        class_dets = dets_by_class.get(c, [])
        # Sort by confidence desc
        class_dets = sorted(class_dets, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(class_dets), dtype=np.float32)
        fp = np.zeros(len(class_dets), dtype=np.float32)

        # Copy matches flags fresh for this computation
        gt_flags = {}
        for fidx, gts in gt_by_class_frame.get(c, {}).items():
            gt_flags[fidx] = [False] * len(gts)

        for i, det in enumerate(class_dets):
            fidx = det['frame']
            det_box = det['bbox']
            gts = gt_by_class_frame.get(c, {}).get(fidx, [])
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gt in enumerate(gts):
                if gt_flags[fidx][gi]:
                    continue
                iou_val = iou(det_box, gt['bbox'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gi
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1.0
                gt_flags[fidx][best_gt_idx] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / float(n_gt)
        precisions = np.divide(tp_cum, (tp_cum + fp_cum + 1e-12))

        # Compute AP using precision envelope method
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        # Identify points where recall changes
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        ap_list.append(ap)

    if not ap_list:
        return None
    return float(np.mean(ap_list))

def parse_ground_truth_sidecar(input_video_path, frame_w, frame_h):
    # Look for a sidecar file with ground truth boxes.
    # Supported candidates:
    #   <input>.gt.txt
    #   <input>.txt
    # Each non-empty, non-comment line format:
    #   frame_index class_id xmin ymin xmax ymax
    # Coordinates can be absolute pixels or normalized [0,1].
    candidates = [
        os.path.splitext(input_video_path)[0] + '.gt.txt',
        os.path.splitext(input_video_path)[0] + '.txt'
    ]
    gt_path = None
    for p in candidates:
        if os.path.isfile(p):
            gt_path = p
            break
    if gt_path is None:
        return {}

    gt_by_frame = {}
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                # allow comma or whitespace separated
                parts = s.replace(',', ' ').split()
                if len(parts) < 6:
                    continue
                try:
                    fidx = int(parts[0])
                    cls = int(parts[1])
                    x1 = float(parts[2]); y1 = float(parts[3]); x2 = float(parts[4]); y2 = float(parts[5])
                except Exception:
                    continue
                # Determine if normalized
                is_normalized = (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0)
                if is_normalized:
                    xmin = int(round(x1 * frame_w))
                    ymin = int(round(y1 * frame_h))
                    xmax = int(round(x2 * frame_w))
                    ymax = int(round(y2 * frame_h))
                else:
                    xmin = int(round(x1))
                    ymin = int(round(y1))
                    xmax = int(round(x2))
                    ymax = int(round(y2))
                xmin, ymin, xmax, ymax = clip_bbox(xmin, ymin, xmax, ymax, frame_w, frame_h)
                gt_by_frame.setdefault(fidx, []).append({
                    'class_id': cls,
                    'bbox': [xmin, ymin, xmax, ymax]
                })
    except Exception as e:
        print("Warning: Failed to parse ground truth file:", e)
        return {}
    return gt_by_frame

# =========================
# Main Pipeline
# =========================
def main():
    # Validate input video
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Prepare output directory
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Initialize interpreter with EdgeTPU
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_indices = get_output_indices(interpreter)
    in_h, in_w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    in_dtype = input_details[0]['dtype']
    in_index = input_details[0]['index']

    # Setup video capture and writer
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video: " + input_path)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open output video for writing: " + output_path)

    # Attempt to load ground-truth boxes if a sidecar file exists
    gt_by_frame_full = parse_ground_truth_sidecar(input_path, frame_w, frame_h)
    has_gt = len(gt_by_frame_full) > 0
    if not has_gt:
        print("Info: No ground-truth sidecar found. mAP will be shown as N/A.")

    # Storage for detections across frames for mAP computation
    all_detections = []
    frame_index = 0

    start_time = time.time()
    frames_processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            inp = preprocess_frame_bgr_to_input(frame, in_w, in_h, in_dtype)

            # Inference
            interpreter.set_tensor(in_index, inp)
            interpreter.invoke()

            boxes_idx, classes_idx, scores_idx, count_idx = output_indices
            boxes = interpreter.get_tensor(interpreter.get_output_details()[boxes_idx]['index'])
            classes = interpreter.get_tensor(interpreter.get_output_details()[classes_idx]['index'])
            scores = interpreter.get_tensor(interpreter.get_output_details()[scores_idx]['index'])
            count = interpreter.get_tensor(interpreter.get_output_details()[count_idx]['index'])

            # Squeeze outputs (expected shapes: [1, N, ...])
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)
            if np.ndim(count) > 0:
                num = int(np.squeeze(count))
            else:
                num = boxes.shape[0]

            # Collect detections above threshold
            detections_this_frame = []
            for i in range(num):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                # Box is [ymin, xmin, ymax, xmax] normalized
                ymin, xmin, ymax, xmax = boxes[i]
                # Convert to absolute pixel coords
                x1 = int(round(xmin * frame_w))
                y1 = int(round(ymin * frame_h))
                x2 = int(round(xmax * frame_w))
                y2 = int(round(ymax * frame_h))
                x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, frame_w, frame_h)
                cls_id = int(classes[i])
                detections_this_frame.append({
                    'frame': frame_index,
                    'class_id': cls_id,
                    'score': score,
                    'bbox': [x1, y1, x2, y2]
                })

            # Draw
            draw_detections(frame, detections_this_frame, labels)

            # Update detection history
            all_detections.extend(detections_this_frame)

            # Compute running mAP if GT available (using GT up to current frame)
            map_text = "mAP: N/A"
            if has_gt:
                # Filter ground-truth up to current frame
                gt_partial = {fi: gt_by_frame_full[fi] for fi in gt_by_frame_full.keys() if fi <= frame_index}
                mAP_val = compute_map(all_detections, gt_partial, iou_threshold=0.5)
                if mAP_val is not None:
                    map_text = f"mAP: {mAP_val * 100.0:.2f}%"

            # Draw mAP on the frame
            cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 255), 2, cv2.LINE_AA)

            # Write frame
            writer.write(frame)

            frame_index += 1
            frames_processed += 1

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start_time
    if elapsed > 0 and frames_processed > 0:
        print(f"Processed {frames_processed} frames in {elapsed:.2f}s ({frames_processed/elapsed:.2f} FPS).")

    # Final mAP over entire video if GT exists
    if has_gt:
        final_mAP = compute_map(all_detections, gt_by_frame_full, iou_threshold=0.5)
        if final_mAP is not None:
            print(f"Final mAP over video: {final_mAP * 100.0:.2f}%")
        else:
            print("Final mAP could not be computed (no ground truth boxes found).")
    else:
        print("No ground truth available; mAP was not computed.")

    print("Output saved to:", output_path)

if __name__ == "__main__":
    main()