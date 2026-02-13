import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: Interpreter must be imported literally as specified
try:
    from ai_edge_litert.interpreter import Interpreter
except Exception as e:
    raise SystemExit("Failed to import ai_edge_litert Interpreter. Ensure the 'ai_edge_litert' package is available. Error: {}".format(e))

def load_labels(label_path):
    labels = []
    if os.path.isfile(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    else:
        # Fallback to an empty list if labels file is missing (script remains runnable)
        labels = []
    return labels

def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

def get_input_size_and_format(input_details):
    shape = input_details[0]['shape']
    # Default to NHWC if possible
    if len(shape) == 4:
        if shape[3] == 3:
            # NHWC
            height, width, channels = int(shape[1]), int(shape[2]), int(shape[3])
            data_format = 'NHWC'
        elif shape[1] == 3:
            # NCHW
            height, width, channels = int(shape[2]), int(shape[3]), int(shape[1])
            data_format = 'NCHW'
        else:
            # Fallback assume NHWC (most TFLite models)
            height, width, channels = int(shape[1]), int(shape[2]), 3
            data_format = 'NHWC'
    else:
        raise SystemExit("Unexpected input tensor shape: {}".format(shape))
    return height, width, channels, data_format

def preprocess_frame(frame_bgr, in_h, in_w, dtype, data_format):
    # Convert BGR to RGB as SSD MobileNet typically expects RGB input
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h))
    input_data = resized
    # Expand to batch and reorder if necessary
    if data_format == 'NHWC':
        input_data = np.expand_dims(input_data, axis=0)
    else:
        # NCHW
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(input_data, axis=0)
    # Convert dtype as required
    if dtype == np.float32:
        input_data = input_data.astype(np.float32)
        # Phase 2.3 Quantization Handling - normalize for floating model
        input_data = (input_data - 127.5) / 127.5
    else:
        input_data = input_data.astype(dtype)
    return input_data

def iou(boxA, boxB):
    # Boxes: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = areaA + areaB - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def cluster_and_mark_tp_fp(dets, iou_threshold=0.5):
    """
    dets: list of dicts: {'score': float, 'box': [xmin, ymin, xmax, ymax]}
    Returns:
      dets_sorted: same det dicts with additional 'is_tp' boolean, sorted by score desc
      n_gt: number of pseudo ground truth objects (clusters)
    """
    if not dets:
        return [], 0
    # Sort detections by score descending
    dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
    clusters = []  # each cluster: {'box': [..], 'members': [det_indices]}
    # Assign detections to clusters; first member of each cluster is TP, others are FP
    for det in dets_sorted:
        assigned = False
        for cl in clusters:
            if iou(det['box'], cl['box']) > iou_threshold:
                det['is_tp'] = False
                cl['members'].append(det)
                assigned = True
                break
        if not assigned:
            det['is_tp'] = True
            clusters.append({'box': det['box'], 'members': [det]})
    n_gt = len(clusters)
    return dets_sorted, n_gt

def compute_ap(dets_sorted, n_gt):
    """
    Compute Average Precision using precision envelope integration.
    dets_sorted: list of det dicts with 'is_tp' key sorted by score desc
    n_gt: number of pseudo ground truth objects
    Returns AP float or None if not computable (n_gt == 0 and no detections)
    """
    if n_gt == 0:
        if len(dets_sorted) == 0:
            return None  # No positives and no predictions: skip for averaging
        else:
            return 0.0   # No ground truth but predictions exist: AP=0
    tp = np.array([1 if d['is_tp'] else 0 for d in dets_sorted], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / float(n_gt)
    # Precision envelope
    # Add boundary points
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Integrate area under curve where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def color_for_class(class_id):
    # Deterministic color from class_id
    np.random.seed(class_id + 12345)
    color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
    # OpenCV uses BGR
    return (int(color[0]), int(color[1]), int(color[2]))

def get_label_name(labels, class_id):
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return "id_{}".format(class_id)

def main():
    # 1.2 Paths/Parameters
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')

    # 1.3 Load Labels if available and relevant
    labels = load_labels(label_path)

    # 1.4 Load Interpreter
    if not os.path.isfile(model_path):
        raise SystemExit("Model file not found at: {}".format(model_path))
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    input_dtype = input_details[0]['dtype']
    in_h, in_w, in_c, data_format = get_input_size_and_format(input_details)
    floating_model = (input_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: single video file
    if not os.path.isfile(input_path):
        raise SystemExit("Input video file not found at: {}".format(input_path))
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit("Failed to open input video: {}".format(input_path))

    # Prepare output writer
    ensure_dir(output_path)
    # Try to use original FPS and size; fallback if unavailable
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        # Read a frame to determine size
        ret, test_frame = cap.read()
        if not ret:
            cap.release()
            raise SystemExit("Could not read frames from video.")
        orig_h, orig_w = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit("Failed to open output video writer at: {}".format(output_path))

    # Variables for output mapping determination (boxes/classes/scores/num_dets)
    mapping_determined = False
    boxes_idx = None
    classes_idx = None
    scores_idx = None
    num_idx = None

    # Variables for mAP accumulation across frames and classes
    per_class_ap_sum = {}
    per_class_ap_count = {}

    frame_index = 0
    start_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # End of video

        # 2.2 Preprocess Data
        input_data = preprocess_frame(frame_bgr, in_h, in_w, input_dtype, data_format)

        # 3.1 Set Input Tensor
        interpreter.set_tensor(input_index, input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling
        # 4.1 Get Output Tensors
        outputs = [interpreter.get_tensor(out['index']) for out in output_details]

        # Determine mapping on first frame using shapes and value ranges
        if not mapping_determined:
            for i, arr in enumerate(outputs):
                a = np.array(arr)
                if a.size == 1:
                    num_idx = i
                elif a.ndim == 3 and a.shape[-1] == 4:
                    boxes_idx = i
            # For classes and scores, both are shape (1, N). Distinguish by value range.
            candidates = [i for i in range(len(outputs)) if i not in (boxes_idx, num_idx)]
            for i in candidates:
                arr = np.array(outputs[i]).astype(np.float32)
                # scores are within [0,1], classes usually > 1
                if np.all(arr >= 0.0) and np.all(arr <= 1.0):
                    scores_idx = i
                else:
                    classes_idx = i
            # Fallback if still ambiguous: try common order [boxes, classes, scores, num]
            if boxes_idx is None or classes_idx is None or scores_idx is None or num_idx is None:
                if len(outputs) >= 4:
                    boxes_idx = 0 if boxes_idx is None else boxes_idx
                    classes_idx = 1 if classes_idx is None else classes_idx
                    scores_idx = 2 if scores_idx is None else scores_idx
                    num_idx = 3 if num_idx is None else num_idx
                else:
                    cap.release()
                    writer.release()
                    raise SystemExit("Could not determine TFLite detection output mappings.")
            mapping_determined = True

        # Extract tensors using determined indices
        boxes = np.array(outputs[boxes_idx])[0]  # shape [N, 4], normalized [ymin, xmin, ymax, xmax]
        classes = np.array(outputs[classes_idx])[0]
        scores = np.array(outputs[scores_idx])[0]
        num_dets = int(np.array(outputs[num_idx]).reshape(-1)[0]) if outputs[num_idx].size >= 1 else len(scores)
        num_dets = min(num_dets, boxes.shape[0], scores.shape[0], classes.shape[0])

        # 4.2 Interpret Results: collect detections per class with bounding boxes and labels
        detections_per_class = {}  # class_id -> list of det dicts {'score': float, 'box': [xmin, ymin, xmax, ymax]}
        drawn_boxes = []  # boxes to draw (top of clusters or all detections), list of tuples (box, label, score)
        for j in range(num_dets):
            score = float(scores[j])
            if score < confidence_threshold:
                continue
            cls_id = int(classes[j])
            # Scale normalized box to pixel coordinates
            ymin, xmin, ymax, xmax = boxes[j]
            xmin = int(max(0, min(orig_w - 1, xmin * orig_w)))
            xmax = int(max(0, min(orig_w - 1, xmax * orig_w)))
            ymin = int(max(0, min(orig_h - 1, ymin * orig_h)))
            ymax = int(max(0, min(orig_h - 1, ymax * orig_h)))
            # 4.3 Post-processing: ensure valid coordinates (clip and correct if needed)
            if xmax < xmin: xmin, xmax = xmax, xmin
            if ymax < ymin: ymin, ymax = ymax, ymin
            box = [xmin, ymin, xmax, ymax]
            # Accumulate detections per class for mAP computation
            if cls_id not in detections_per_class:
                detections_per_class[cls_id] = []
            detections_per_class[cls_id].append({'score': score, 'box': box})
            # For drawing: collect now, actual selection (top of clusters) will be determined below
            drawn_boxes.append((box, cls_id, score))

        # Compute per-class AP for this frame using pseudo GT via clustering duplicates
        frame_ap_per_class = {}
        boxes_to_draw = []  # top-of-cluster boxes to draw to avoid duplicates
        for cls_id, dets in detections_per_class.items():
            dets_sorted, n_gt = cluster_and_mark_tp_fp(dets, iou_threshold=0.5)
            ap = compute_ap(dets_sorted, n_gt)
            if ap is not None:
                frame_ap_per_class[cls_id] = ap
                # Update running averages
                per_class_ap_sum[cls_id] = per_class_ap_sum.get(cls_id, 0.0) + ap
                per_class_ap_count[cls_id] = per_class_ap_count.get(cls_id, 0) + 1
            # For drawing: draw only top of each cluster (is_tp True)
            for det in dets_sorted:
                if det.get('is_tp', False):
                    boxes_to_draw.append((det['box'], cls_id, det['score']))

        # Compute running mAP across classes seen so far
        running_map = 0.0
        cls_with_stats = [cid for cid, cnt in per_class_ap_count.items() if cnt > 0]
        if len(cls_with_stats) > 0:
            per_class_avgs = [per_class_ap_sum[cid] / per_class_ap_count[cid] for cid in cls_with_stats]
            running_map = float(np.mean(per_class_avgs))

        # 4.4 Handle Output: Draw results and mAP onto the frame and write to output video
        # Draw rectangles and labels
        for box, cls_id, score in boxes_to_draw:
            xmin, ymin, xmax, ymax = box
            color = color_for_class(cls_id)
            cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, 2)
            label = get_label_name(labels, cls_id)
            caption = "{}: {:.2f}".format(label, score)
            # Text background
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, ymin), color, -1)
            cv2.putText(frame_bgr, caption, (xmin + 2, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Overlay running mAP on the frame
        map_text = "mAP: {:.3f}".format(running_map)
        cv2.rectangle(frame_bgr, (5, 5), (5 + 200, 5 + 30), (0, 0, 0), -1)
        cv2.putText(frame_bgr, map_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Write frame to output video
        writer.write(frame_bgr)

        frame_index += 1

    # Phase 5: Cleanup
    cap.release()
    writer.release()

    # Print final mAP to console
    if len([cid for cid, cnt in per_class_ap_count.items() if cnt > 0]) > 0:
        per_class_avgs = {cid: (per_class_ap_sum[cid] / per_class_ap_count[cid]) for cid in per_class_ap_count if per_class_ap_count[cid] > 0}
        final_map = float(np.mean(list(per_class_avgs.values())))
        print("Final mAP over processed video (proxy via duplicate clustering): {:.4f}".format(final_map))
        # Optionally print per-class AP averages
        for cid, avg_ap in sorted(per_class_avgs.items(), key=lambda x: x[0]):
            print("Class {} ({}): AP_avg = {:.4f}".format(cid, get_label_name(labels, cid), avg_ap))
    else:
        print("No detections above threshold were found; mAP could not be computed.")

if __name__ == "__main__":
    main()