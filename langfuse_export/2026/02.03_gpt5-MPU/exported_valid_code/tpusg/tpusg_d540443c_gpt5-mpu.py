import os
import time
import numpy as np
import cv2

# =========================
# Phase 1: Setup
# =========================

# 1.1 Imports: TFLite Interpreter and EdgeTPU delegate with fallback imports
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_source = "tflite_runtime"
except ImportError:
    try:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
        tflite_source = "tensorflow.lite"
    except Exception as e:
        print("ERROR: Unable to import TFLite Interpreter. Ensure tflite_runtime or TensorFlow Lite is installed.")
        raise

# 1.2 Paths/Parameters (from CONFIGURATION PARAMETERS)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5
pseudo_gt_high_conf_threshold = 0.75  # threshold to create pseudo ground-truth from high-confidence detections
iou_threshold = 0.5  # IoU threshold for a correct match in mAP computation

# 1.3 Load Labels (if provided and needed)
def load_labels(path):
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # If labels may contain "id label" format, split and take the last token(s)
                # Given snippet shows plain names per line; we keep simple
                labels.append(line)
    except Exception as e:
        print(f"WARNING: Failed to load labels from {path}. Error: {e}")
        labels = []
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU delegate
def make_interpreter_with_edgetpu(model_path_local):
    last_error = None
    for libpath in ['libedgetpu.so.1.0', '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0']:
        try:
            interpreter_local = Interpreter(
                model_path=model_path_local,
                experimental_delegates=[load_delegate(libpath)]
            )
            return interpreter_local
        except Exception as e:
            last_error = e
    raise RuntimeError(f"ERROR: Failed to load EdgeTPU delegate. Last error: {last_error}. "
                       f"Ensure the EdgeTPU runtime is installed and the Coral is connected.")

# Check files
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input video file not found: {input_path}")
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Create interpreter
try:
    interpreter = make_interpreter_with_edgetpu(model_path)
except Exception as e:
    print(str(e))
    raise SystemExit(1)

# Allocate tensors
try:
    interpreter.allocate_tensors()
except Exception as e:
    print(f"ERROR: Failed to allocate tensors: {e}")
    raise SystemExit(1)

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine input shape and dtype
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']  # typically [1, height, width, 3]
input_height, input_width = int(input_shape[1]), int(input_shape[2])
input_dtype = input_details[0]['dtype']
floating_model = (input_dtype == np.float32)

# Helper: color per class for drawing
def color_for_class(cid):
    np.random.seed(cid + 7)
    color = np.random.randint(0, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

# =========================
# Phase 2: Input Acquisition & Preprocessing Loop
# =========================

# 2.1 Acquire Input Data: open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"ERROR: Cannot open input video: {input_path}")
    raise SystemExit(1)

# Determine video properties for output writer
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1e-3:
    fps = 30.0  # fallback
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
if not out_writer.isOpened():
    print(f"ERROR: Cannot open output video for writing: {output_path}")
    cap.release()
    raise SystemExit(1)

def preprocess_frame(frame_bgr):
    # 2.2 Preprocess Data: resize and normalize as needed based on input_details
    frame_resized = cv2.resize(frame_bgr, (input_width, input_height))
    # Convert BGR to RGB as TFLite models typically expect RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb, axis=0)
    if floating_model:
        # 2.3 Quantization Handling: normalize to [-1, 1] for floating models
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # For quantized models (uint8), pass as-is
        if input_dtype == np.uint8:
            input_data = np.uint8(input_data)
        else:
            # Fallback: cast to required dtype without normalization
            input_data = input_data.astype(input_dtype)
    return input_data

# =========================
# Utilities for Detection and mAP
# =========================

def interpret_outputs(interpreter_obj, output_details_list):
    # 4.1 Get Output Tensor(s)
    outputs = [interpreter_obj.get_tensor(od['index']) for od in output_details_list]
    # Attempt to map outputs by common SSD order: [boxes, classes, scores, count]
    # Many EdgeTPU detection models follow this order.
    if len(outputs) >= 4:
        boxes = outputs[0]
        classes = outputs[1]
        scores = outputs[2]
        count = outputs[3]
    else:
        # If unusual, best-effort mapping by shapes
        boxes, classes, scores, count = None, None, None, None
        for out, det in zip(outputs, output_details_list):
            shp = det['shape']
            if len(shp) == 3 and shp[-1] == 4:
                boxes = out
            elif len(shp) == 2 and shp[1] > 1 and det['dtype'] in [np.float32, np.int64, np.int32, np.uint8]:
                # Could be classes or scores; we check dtype and value ranges later
                if classes is None:
                    classes = out
                else:
                    scores = out
            elif len(shp) == 1 and shp[0] == 1:
                count = out
        if boxes is None or classes is None or scores is None or count is None:
            raise RuntimeError("Unexpected model output format; cannot interpret detection outputs.")
    return boxes, classes, scores, count

def clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2

def iou(boxA, boxB):
    # boxes: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-10)

def compute_map(predictions, ground_truths, num_classes, iou_thr=0.5):
    # predictions: list of dicts {class_id, score, bbox:[x1,y1,x2,y2], frame_id}
    # ground_truths: list of dicts {class_id, bbox:[x1,y1,x2,y2], frame_id}
    # Returns: mAP (float or None if no GT)
    # Prepare GT structures
    gts_by_class_frame = {}
    for gt in ground_truths:
        cid = gt['class_id']
        fid = gt['frame_id']
        gts_by_class_frame.setdefault(cid, {}).setdefault(fid, []).append(gt['bbox'])
    # For each class, compute AP
    ap_list = []
    for cid in range(num_classes):
        preds_c = [p for p in predictions if p['class_id'] == cid]
        if cid not in gts_by_class_frame:
            continue
        total_gt = sum(len(bxs) for bxs in gts_by_class_frame[cid].values())
        if total_gt == 0:
            continue
        # sort predictions by confidence descending
        preds_c_sorted = sorted(preds_c, key=lambda d: d['score'], reverse=True)
        # matched flags for GT
        matched_flags = {fid: [False] * len(gts_by_class_frame[cid][fid]) for fid in gts_by_class_frame[cid]}
        tp = np.zeros(len(preds_c_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_c_sorted), dtype=np.float32)
        for i, p in enumerate(preds_c_sorted):
            fid = p['frame_id']
            bbox_p = p['bbox']
            best_iou = 0.0
            best_gt_idx = -1
            if fid in gts_by_class_frame[cid]:
                gt_boxes = gts_by_class_frame[cid][fid]
                for j, gt_box in enumerate(gt_boxes):
                    if matched_flags[fid][j]:
                        continue
                    iou_val = iou(bbox_p, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = j
            if best_iou >= iou_thr and best_gt_idx >= 0:
                tp[i] = 1.0
                matched_flags[fid][best_gt_idx] = True
            else:
                fp[i] = 1.0
        if len(tp) == 0:
            continue
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        recall = cum_tp / float(total_gt)
        # 11-point interpolated AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recall >= t):
                p = np.max(precision[recall >= t])
            else:
                p = 0.0
            ap += p / 11.0
        ap_list.append(ap)
    if len(ap_list) == 0:
        return None
    return float(np.mean(ap_list))

def draw_detections(frame, detections_to_draw, labels_list, map_value):
    # Draw detections and mAP value on frame
    for det in detections_to_draw:
        x1, y1, x2, y2 = det['bbox']
        cid = det['class_id']
        score = det['score']
        color = color_for_class(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if 0 <= cid < len(labels_list):
            label_name = labels_list[cid]
        else:
            label_name = f"id:{cid}"
        text = f"{label_name} {score:.2f}"
        # Text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Draw mAP
    if map_value is None:
        map_text = "mAP: N/A"
    else:
        map_text = f"mAP: {map_value:.3f}"
    cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

# =========================
# Phase 3: Inference and Phase 4: Output Interpretation & Handling
# =========================

running_predictions = []  # list of dicts: {class_id, score, bbox:[x1,y1,x2,y2], frame_id}
running_ground_truths = []  # pseudo GT from high-confidence detections

frame_index = 0
start_time = time.time()

try:
    while True:
        # 2.4 Loop Control: read next frame; break if none
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_data = preprocess_frame(frame)

        # Phase 3.1: Set Input Tensor
        interpreter.set_tensor(input_index, input_data)

        # Phase 3.2: Run Inference
        interpreter.invoke()

        # Phase 4.1: Get outputs
        boxes, classes, scores, count = interpret_outputs(interpreter, output_details)

        # Convert outputs to numpy arrays
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        if np.isscalar(count):
            num = int(count)
        elif isinstance(count, np.ndarray):
            num = int(np.squeeze(count))
        else:
            num = len(scores)

        # 4.2 Interpret Results and 4.3 Post-processing:
        detections_for_draw = []
        predictions_this_frame = []
        gts_this_frame = []

        for i in range(num):
            score = float(scores[i])
            cid = int(classes[i])
            # The model returns boxes in normalized [ymin, xmin, ymax, xmax]
            y1_norm, x1_norm, y2_norm, x2_norm = boxes[i]
            x1 = int(x1_norm * frame_width)
            y1 = int(y1_norm * frame_height)
            x2 = int(x2_norm * frame_width)
            y2 = int(y2_norm * frame_height)
            # Ensure x1 <= x2, y1 <= y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            # 4.3 Bounding box clipping
            x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, frame_width, frame_height)

            # Collect detections for drawing if above main confidence threshold
            if score >= confidence_threshold:
                detections_for_draw.append({
                    'class_id': cid,
                    'score': score,
                    'bbox': [x1, y1, x2, y2]
                })
                predictions_this_frame.append({
                    'class_id': cid,
                    'score': score,
                    'bbox': [x1, y1, x2, y2],
                    'frame_id': frame_index
                })

            # Collect pseudo ground-truth using higher confidence threshold
            if score >= pseudo_gt_high_conf_threshold:
                gts_this_frame.append({
                    'class_id': cid,
                    'bbox': [x1, y1, x2, y2],
                    'frame_id': frame_index
                })

        # Update running lists
        running_predictions.extend(predictions_this_frame)
        running_ground_truths.extend(gts_this_frame)

        # Compute running mAP across frames processed so far
        num_classes = max(len(labels), int(np.max(classes)) + 1 if classes.size > 0 else 1)
        current_map = compute_map(running_predictions, running_ground_truths, num_classes, iou_thr=iou_threshold)

        # 4.4 Handle Output: draw and write frame
        frame_out = draw_detections(frame, detections_for_draw, labels, current_map)
        out_writer.write(frame_out)

        frame_index += 1

except Exception as e:
    print(f"ERROR during processing: {e}")
finally:
    # =========================
    # Phase 5: Cleanup
    # =========================
    cap.release()
    out_writer.release()

# Print summary
elapsed = time.time() - start_time
print(f"Processed {frame_index} frames in {elapsed:.2f}s ({(frame_index / max(elapsed,1e-6)):.2f} FPS).")
# Final mAP computation
final_num_classes = len(labels) if len(labels) > 0 else 90
final_map = compute_map(running_predictions, running_ground_truths, final_num_classes, iou_thr=iou_threshold)
if final_map is None:
    print("Final mAP: N/A (no pseudo ground-truth was collected; increase pseudo_gt_high_conf_threshold if needed).")
else:
    print(f"Final mAP (pseudo-GT based): {final_map:.4f}")
print(f"Output video saved to: {output_path}")