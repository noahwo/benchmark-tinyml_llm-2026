import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =========================
# Utilities
# =========================
def load_labels(path):
    """
    Load labels from a label map file.
    Supports:
      - Simple list (each line a label; may include '???' for background)
      - PBTXT-like format with 'item { id: X name: 'label' }'
    Returns a dict: {id(int): name(str)}
    """
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Detect PBTXT-like structure
    if 'item' in content and 'id' in content and 'name' in content:
        # Simple manual parse without regex
        lines = [ln.strip() for ln in content.splitlines()]
        current_id = None
        current_name = None
        for ln in lines:
            if ln.startswith('item'):
                current_id = None
                current_name = None
            if 'id' in ln and ':' in ln:
                # Extract digits for id
                digits = ''.join([ch for ch in ln if ch.isdigit()])
                if digits:
                    try:
                        current_id = int(digits)
                    except Exception:
                        current_id = None
            if 'name' in ln and ':' in ln:
                # Try to extract between single or double quotes
                name = None
                if "'" in ln:
                    try:
                        name = ln.split("'", 2)[1]
                    except Exception:
                        name = None
                if name is None and '"' in ln:
                    try:
                        name = ln.split('"', 2)[1]
                    except Exception:
                        name = None
                current_name = name
            if ln.endswith('}') and current_id is not None and current_name is not None:
                labels[current_id] = current_name
                current_id = None
                current_name = None
        if labels:
            return labels
    # Fallback: simple list
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    for idx, name in enumerate(lines):
        labels[idx] = name
    return labels

def get_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame, input_shape, input_dtype):
    """
    Resize and convert color to match model input.
    input_shape expected as (1, height, width, channels) or similar.
    """
    if len(input_shape) == 4:
        height, width = int(input_shape[1]), int(input_shape[2])
    elif len(input_shape) == 3:
        height, width = int(input_shape[0]), int(input_shape[1])
    else:
        raise ValueError("Unexpected input tensor shape: {}".format(input_shape))
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    input_data = resized
    if input_dtype == np.float32:
        input_data = (input_data.astype(np.float32) / 255.0).astype(np.float32)
    else:
        input_data = input_data.astype(input_dtype)
    # Add batch dimension if needed
    if len(input_shape) == 4:
        input_data = np.expand_dims(input_data, axis=0)
    return input_data

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Try to map outputs by typical SSD tensors: boxes, classes, scores, num
    boxes = None
    classes = None
    scores = None
    num = None
    for out in output_details:
        idx = out['index']
        tensor = interpreter.get_tensor(idx)
        shape = tensor.shape
        # Heuristics based on shapes and dtypes
        if tensor.ndim == 3 and shape[-1] == 4:
            boxes = tensor
        elif tensor.ndim == 2 and (tensor.shape[-1] > 1) and np.issubdtype(tensor.dtype, np.floating):
            # Could be classes or scores; check value ranges
            if np.all((tensor >= 0.0) & (tensor <= 1.0)):
                scores = tensor
            else:
                classes = tensor
        elif tensor.ndim == 2 and np.issubdtype(tensor.dtype, np.integer):
            classes = tensor
        elif tensor.size == 1:
            num = int(np.squeeze(tensor).tolist())
    # Some models return [1, N, 4], [1, N], [1, N], [1]
    if boxes is not None and boxes.ndim == 3:
        boxes = boxes[0]
    if classes is not None and classes.ndim == 2:
        classes = classes[0]
    if scores is not None and scores.ndim == 2:
        scores = scores[0]
    if num is None and boxes is not None:
        num = boxes.shape[0]
    return boxes, classes, scores, num

def denormalize_boxes(boxes, frame_width, frame_height):
    """
    Convert normalized ymin, xmin, ymax, xmax to pixel coordinates [xmin, ymin, xmax, ymax]
    """
    pixel_boxes = []
    for b in boxes:
        ymin, xmin, ymax, xmax = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        x1 = max(0, min(frame_width - 1, int(xmin * frame_width)))
        y1 = max(0, min(frame_height - 1, int(ymin * frame_height)))
        x2 = max(0, min(frame_width - 1, int(xmax * frame_width)))
        y2 = max(0, min(frame_height - 1, int(ymax * frame_height)))
        pixel_boxes.append([x1, y1, x2, y2])
    return pixel_boxes

def draw_detections(frame, boxes, classes, scores, labels, conf_thr):
    h, w = frame.shape[:2]
    count = 0
    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf_thr:
            continue
        count += 1
        cls_id = int(classes[i]) if classes is not None else -1
        label = labels.get(cls_id, str(cls_id))
        # color derived from class id
        color = ((37 * (cls_id + 1)) % 255, (17 * (cls_id + 2)) % 255, (29 * (cls_id + 3)) % 255)
        # box is normalized; convert to pixel
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = int(max(0, min(w - 1, xmin * w)))
        y1 = int(max(0, min(h - 1, ymin * h)))
        x2 = int(max(0, min(w - 1, xmax * w)))
        y2 = int(max(0, min(h - 1, ymax * h)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = "{}: {:.2f}".format(label, score)
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return count

def iou(box_a, box_b):
    """
    box: [xmin, ymin, xmax, ymax]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = ax1 if ax1 > bx1 else bx1
    inter_y1 = ay1 if ay1 > by1 else by1
    inter_x2 = ax2 if ax2 < bx2 else bx2
    inter_y2 = ay2 if ay2 < by2 else by2
    iw = inter_x2 - inter_x1 + 1
    ih = inter_y2 - inter_y1 + 1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)

def compute_map(predictions_by_class, ground_truth_by_class, iou_threshold=0.5):
    """
    Compute mAP across classes with available ground truth.
    predictions_by_class: {class_id: [(image_id, score, [xmin, ymin, xmax, ymax]), ...]}
    ground_truth_by_class: {class_id: {image_id: [[xmin, ymin, xmax, ymax], ...], ...}, ...}
    Returns mAP (float). If no GT is available, returns 0.0.
    """
    ap_list = []
    for cls_id in predictions_by_class:
        preds = predictions_by_class.get(cls_id, [])
        gts_dict = ground_truth_by_class.get(cls_id, {})
        total_gts = 0
        for img_id in gts_dict:
            total_gts += len(gts_dict[img_id])
        # Skip classes with no GT
        if total_gts == 0:
            continue
        # Prepare matched flags for GTs
        matched = {}
        for img_id in gts_dict:
            matched[img_id] = [False] * len(gts_dict[img_id])
        # Sort predictions by descending score
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        tps = []
        fps = []
        for (img_id, score, box_p) in preds_sorted:
            if img_id in gts_dict:
                gt_boxes = gts_dict[img_id]
                gt_matched_flags = matched[img_id]
            else:
                gt_boxes = []
                gt_matched_flags = []
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gt_box in enumerate(gt_boxes):
                if gt_matched_flags[gi]:
                    continue
                ov = iou(box_p, gt_box)
                if ov > best_iou:
                    best_iou = ov
                    best_gt_idx = gi
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched_flags[best_gt_idx] = True
                tps.append(1.0)
                fps.append(0.0)
            else:
                tps.append(0.0)
                fps.append(1.0)
        if len(tps) == 0:
            ap_list.append(0.0)
            continue
        tp_cum = np.cumsum(np.array(tps, dtype=np.float32))
        fp_cum = np.cumsum(np.array(fps, dtype=np.float32))
        recalls = tp_cum / float(total_gts)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        # Compute AP using 101-point interpolation
        recall_levels = np.linspace(0.0, 1.0, 101)
        precision_interpolated = []
        for r in recall_levels:
            mask = recalls >= r
            if np.any(mask):
                precision_interpolated.append(np.max(precisions[mask]))
            else:
                precision_interpolated.append(0.0)
        ap = float(np.mean(np.array(precision_interpolated, dtype=np.float32)))
        ap_list.append(ap)
    if len(ap_list) == 0:
        return 0.0
    return float(np.mean(np.array(ap_list, dtype=np.float32)))

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# =========================
# Main application
# =========================
def main():
    # Load labels
    labels = load_labels(label_path)

    # Setup interpreter
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: {}".format(model_path))
    interpreter = get_interpreter(model_path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Setup video IO
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input video file not found: {}".format(input_path))
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: {}".format(input_path))

    # Try to read first frame to get properties reliably
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Failed to read frames from video: {}".format(input_path))

    height, width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter for: {}".format(output_path))

    # mAP bookkeeping
    predictions_by_class = {}  # {cls_id: [(frame_idx, score, [x1,y1,x2,y2]), ...]}
    ground_truth_by_class = {}  # No ground truth provided; remains empty
    frame_idx = 0

    # Process first frame then loop remaining
    frames_to_process = [first_frame]

    def process_frame(frame, frame_idx):
        nonlocal predictions_by_class
        # Prepare input
        input_data = preprocess_frame(frame, input_shape, input_dtype)
        # Inference
        boxes, classes, scores, num = run_inference(interpreter, input_data)
        if boxes is None or scores is None:
            # Cannot proceed if outputs missing
            det_count = 0
            map_val = compute_map(predictions_by_class, ground_truth_by_class, iou_threshold=0.5)
            cv2.putText(frame, "mAP: {:.3f}".format(map_val), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            return frame, det_count
        n = int(num) if num is not None else len(scores)
        n = min(n, len(scores), len(boxes))
        # Draw detections and accumulate for mAP with pixel boxes
        det_count = 0
        h, w = frame.shape[:2]
        for i in range(n):
            sc = float(scores[i])
            if sc < confidence_threshold:
                continue
            det_count += 1
            cls_id = int(classes[i]) if classes is not None else -1
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(max(0, min(w - 1, xmin * w)))
            y1 = int(max(0, min(h - 1, ymin * h)))
            x2 = int(max(0, min(w - 1, xmax * w)))
            y2 = int(max(0, min(h - 1, ymax * h)))
            # Accumulate prediction
            if cls_id not in predictions_by_class:
                predictions_by_class[cls_id] = []
            predictions_by_class[cls_id].append((frame_idx, sc, [x1, y1, x2, y2]))
        # Draw on frame
        draw_detections(frame, boxes[:n], classes[:n] if classes is not None else None, scores[:n], labels, confidence_threshold)
        # Compute running mAP (will be 0.0 without GT)
        map_val = compute_map(predictions_by_class, ground_truth_by_class, iou_threshold=0.5)
        cv2.putText(frame, "mAP: {:.3f}".format(map_val), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 2, cv2.LINE_AA)
        return frame, det_count

    total_frames = 0
    total_dets = 0
    t0 = time.time()

    # Process first frame
    frame_out, dets = process_frame(frames_to_process[0], frame_idx)
    writer.write(frame_out)
    total_frames += 1
    total_dets += dets
    frame_idx += 1

    # Process remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_out, dets = process_frame(frame, frame_idx)
        writer.write(frame_out)
        total_frames += 1
        total_dets += dets
        frame_idx += 1

    # Cleanup
    cap.release()
    writer.release()
    elapsed = time.time() - t0

    # Final mAP over processed video (will be 0.0 if no GT provided)
    final_map = compute_map(predictions_by_class, ground_truth_by_class, iou_threshold=0.5)

    print("Processing complete.")
    print("Input video: {}".format(input_path))
    print("Output video: {}".format(output_path))
    print("Frames processed: {}".format(total_frames))
    print("Total detections (score >= {}): {}".format(confidence_threshold, total_dets))
    print("Final mAP (IoU=0.5): {:.3f}".format(final_map))
    if total_frames > 0 and elapsed > 0:
        print("Average FPS: {:.2f}".format(total_frames / elapsed))

if __name__ == "__main__":
    main()