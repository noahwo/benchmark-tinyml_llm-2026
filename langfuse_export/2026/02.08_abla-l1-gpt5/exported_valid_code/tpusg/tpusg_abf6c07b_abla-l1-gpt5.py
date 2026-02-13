import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration parameters
# =========================
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

# Optional ground-truth file (plain text) for mAP computation.
# Expected line format (comma-separated, 6 values per line):
# frame_index, class_id_or_name, xmin, ymin, xmax, ymax
# Coordinates in absolute pixel units, using the original video frame size.
gt_path = os.path.splitext(input_path)[0] + "_gt.txt"

# =========================
# Utility functions
# =========================
def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if parts[0].isdigit():
                idx = int(parts[0])
                name = parts[1] if len(parts) > 1 else str(idx)
                labels[idx] = name
            else:
                # If no id is provided, enumerate by line index
                labels[i] = line
    return labels

def set_input_tensor(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensors(interpreter):
    # Assumes standard TFLite SSD detection model outputs:
    # [boxes, classes, scores, num_detections]
    output_details = interpreter.get_output_details()
    def dequantize(o_detail, data):
        # If quantized, dequantize; else return float data
        if 'quantization' in o_detail and o_detail['quantization'] and o_detail['quantization'][0] != 0:
            scale, zero_point = o_detail['quantization']
            return scale * (data.astype(np.float32) - zero_point)
        return data.astype(np.float32)

    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    count = interpreter.get_tensor(output_details[3]['index'])

    boxes = dequantize(output_details[0], boxes)[0]          # [N, 4] in normalized [ymin, xmin, ymax, xmax]
    classes = dequantize(output_details[1], classes)[0]      # [N]
    scores = dequantize(output_details[2], scores)[0]        # [N]
    num = int(dequantize(output_details[3], count)[0])       # scalar

    return boxes, classes, scores, num

def preprocess_frame(frame, input_shape, input_dtype):
    # Convert BGR to RGB and resize to model input shape
    h, w = input_shape[1], input_shape[2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))
    if input_dtype == np.float32:
        # Normalize to [0,1]
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    # Add batch dimension
    return np.expand_dims(input_data, axis=0)

def draw_detections(frame, detections, labels):
    # detections: list of dicts with keys: bbox [xmin,ymin,xmax,ymax], class_id, score
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        score = det['score']
        label = labels.get(class_id, str(class_id))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = "{}: {:.2f}".format(label, score)
        # Draw filled background for text
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def put_map_text(frame, map_text):
    # Put mAP text on the top-left corner with background
    text = map_text
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (5, 5), (5 + tw + 10, 5 + th + baseline + 10), (50, 50, 50), -1)
    cv2.putText(frame, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def iou_xyxy(a, b):
    # a, b: [xmin, ymin, xmax, ymax]
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def compute_map_running(dets_by_class, gt_by_class_and_frame, iou_thresh, max_frame_idx):
    # dets_by_class: dict[class_id] -> list of (frame_idx, score, bbox)
    # gt_by_class_and_frame: dict[class_id] -> dict[frame_idx] -> list of bboxes
    # Evaluate mAP across classes with available GT up to max_frame_idx
    aps = []
    for class_id in sorted(set(list(dets_by_class.keys()) + list(gt_by_class_and_frame.keys()))):
        # Collect detections up to current frame
        dets = dets_by_class.get(class_id, [])
        dets = [d for d in dets if d[0] <= max_frame_idx]
        if not dets and class_id not in gt_by_class_and_frame:
            continue

        # Build GT pool up to current frame
        gt_frames = gt_by_class_and_frame.get(class_id, {})
        gt_pool = {}
        npos = 0
        for fidx, gt_list in gt_frames.items():
            if fidx <= max_frame_idx:
                gt_pool[fidx] = {'boxes': list(gt_list), 'matched': [False] * len(gt_list)}
                npos += len(gt_list)

        if npos == 0:
            # No GT for this class up to current frame; skip AP for this class
            continue

        # Sort detections by score descending
        dets_sorted = sorted(dets, key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(dets_sorted), dtype=np.float32)
        fp = np.zeros(len(dets_sorted), dtype=np.float32)

        for i, (fidx, score, box) in enumerate(dets_sorted):
            gtf = gt_pool.get(fidx, None)
            if gtf is None or not gtf['boxes']:
                fp[i] = 1.0
                continue

            # Find best IoU match among GT boxes not yet matched
            ious = [iou_xyxy(box, gt_box) for gt_box in gtf['boxes']]
            best_i = -1
            best_iou = 0.0
            for j, val in enumerate(ious):
                if val > best_iou and not gtf['matched'][j]:
                    best_iou = val
                    best_i = j

            if best_i >= 0 and best_iou >= iou_thresh:
                tp[i] = 1.0
                gtf['matched'][best_i] = True
            else:
                fp[i] = 1.0

        # Precision-Recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / float(npos)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        # AP using precision envelope
        # Add boundary points
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Sum over recall changes
        idxs = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idxs + 1] - mrec[idxs]) * mpre[idxs + 1])
        aps.append(ap)

    if not aps:
        return None
    return float(np.mean(aps))

def load_ground_truth_txt(gt_txt_path, labels_map, frame_width, frame_height):
    # Returns dict[class_id] -> dict[frame_idx] -> list of [xmin,ymin,xmax,ymax]
    # labels_map: dict[id] -> name
    inverse_labels = {v: k for k, v in labels_map.items()}
    gt_by_class_and_frame = {}
    if not os.path.isfile(gt_txt_path):
        return None

    with open(gt_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != 6:
                # Skip malformed lines
                continue
            try:
                fidx_str, cls_str, x1_str, y1_str, x2_str, y2_str = parts
                fidx = int(fidx_str)
                # class id or name
                if cls_str.isdigit():
                    cid = int(cls_str)
                else:
                    cid = inverse_labels.get(cls_str, None)
                    if cid is None:
                        # Unknown class name; skip
                        continue
                x1 = int(float(x1_str))
                y1 = int(float(y1_str))
                x2 = int(float(x2_str))
                y2 = int(float(y2_str))
                # Clamp to frame bounds
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(0, min(x2, frame_width - 1))
                y2 = max(0, min(y2, frame_height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                if cid not in gt_by_class_and_frame:
                    gt_by_class_and_frame[cid] = {}
                if fidx not in gt_by_class_and_frame[cid]:
                    gt_by_class_and_frame[cid][fidx] = []
                gt_by_class_and_frame[cid][fidx].append([x1, y1, x2, y2])
            except Exception:
                # Skip parsing errors
                continue
    return gt_by_class_and_frame

# =========================
# Main processing
# =========================
def main():
    # 1) Setup, load interpreter with EdgeTPU delegate
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e:
        print("Failed to load EdgeTPU delegate or model:", e)
        return

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details['shape']  # [1, H, W, C]
    input_dtype = input_details['dtype']

    # Load labels
    labels = load_labels(label_path)
    if not labels:
        print("Warning: No labels were loaded or label file missing. Will use class IDs.")

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open input video:", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer (MP4 with H264/MP4V depending on availability)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: Cannot open output video for writing:", output_path)
        cap.release()
        return

    # Load optional ground truth for mAP
    gt_by_class_and_frame = load_ground_truth_txt(gt_path, labels, width, height)
    has_gt = gt_by_class_and_frame is not None
    if has_gt:
        print("Ground-truth file found for mAP:", gt_path)
    else:
        print("No ground-truth file found; mAP will be shown as N/A.")
    iou_threshold = 0.5

    # For running mAP calculation
    # dets_by_class: class_id -> list of (frame_idx, score, [xmin,ymin,xmax,ymax])
    dets_by_class = {}

    frame_index = -1
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        # 2) Preprocess
        input_tensor = preprocess_frame(frame, input_shape, input_dtype)
        set_input_tensor(interpreter, input_tensor)

        # 3) Inference
        interpreter.invoke()

        # 4) Output handling: parse detections
        boxes, classes, scores, num = get_output_tensors(interpreter)

        detections = []
        # Convert normalized boxes to pixel coordinates and filter by confidence
        for i in range(num):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            class_id = int(classes[i])
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(max(0, min(xmin * width, width - 1)))
            y1 = int(max(0, min(ymin * height, height - 1)))
            x2 = int(max(0, min(xmax * width, width - 1)))
            y2 = int(max(0, min(ymax * height, height - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'score': score
            })

            # Accumulate for mAP
            if class_id not in dets_by_class:
                dets_by_class[class_id] = []
            dets_by_class[class_id].append((frame_index, score, [x1, y1, x2, y2]))

        # Draw detections on frame
        draw_detections(frame, detections, labels)

        # Compute running mAP if GT available
        if has_gt:
            mAP = compute_map_running(dets_by_class, gt_by_class_and_frame, iou_threshold, frame_index)
            if mAP is None:
                map_text = "mAP: N/A (no GT in current range)"
            else:
                map_text = "mAP@IoU{:.2f}: {:.2f}%".format(iou_threshold, mAP * 100.0)
        else:
            map_text = "mAP: N/A (no ground truth)"

        # Put mAP text on frame
        put_map_text(frame, map_text)

        # Write to output
        out.write(frame)

    cap.release()
    out.release()
    elapsed = time.time() - t0

    # Final report
    if has_gt:
        # Compute final mAP across all frames processed
        final_mAP = compute_map_running(dets_by_class, gt_by_class_and_frame, iou_threshold, frame_index)
        if final_mAP is None:
            print("Final mAP: N/A (no matching ground truth)")
        else:
            print("Final mAP@IoU{:.2f}: {:.2f}%".format(iou_threshold, final_mAP * 100.0))
    else:
        print("Final mAP: N/A (no ground truth file)")

    print("Processed {} frames in {:.2f}s. Output saved to: {}".format(frame_index + 1, elapsed, output_path))


if __name__ == "__main__":
    main()