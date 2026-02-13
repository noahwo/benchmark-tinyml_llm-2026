import os
import time
import json
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# =========================
# CONFIGURATION PARAMETERS
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

# Optional ground truth path for mAP calculation (if exists, it will be used)
# Expected JSON formats supported (any one):
# 1) {"frames":[{"frame_index":0,"objects":[{"bbox":[xmin,ymin,xmax,ymax],"class":"sheep"}]}, ...]}
# 2) {"0":[{"bbox":[xmin,ymin,xmax,ymax],"class":1}, ...], "1":[...], ...}
# 3) [{"frame_index":0,"objects":[...]} , ...]
POSSIBLE_GT_PATHS = [
    os.path.join(os.path.dirname(INPUT_PATH), "sheeps_gt.json"),
    os.path.join(os.path.dirname(INPUT_PATH), "sheeps_annotations.json"),
]


# =========================
# HELPER FUNCTIONS
# =========================
def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name == "":
                continue
            labels[i] = name
    return labels


def init_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def prepare_input(frame_bgr, input_details):
    # Assumes input tensor is NHWC
    in_shape = input_details[0]['shape']
    in_dtype = input_details[0]['dtype']
    height, width = int(in_shape[1]), int(in_shape[2])

    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if in_dtype == np.float32:
        # Normalize to [0,1]
        input_data = (rgb.astype(np.float32) / 255.0).reshape(1, height, width, 3)
    else:
        input_data = rgb.astype(in_dtype).reshape(1, height, width, 3)
    return input_data


def get_outputs(interpreter, output_details):
    # Identify outputs by their shapes/types
    # Expecting: boxes [1, N, 4], classes [1, N], scores [1, N], count [1]
    outs = {}
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        shape = data.shape
        if len(shape) == 3 and shape[-1] == 4:
            outs['boxes'] = data[0]
        elif len(shape) == 2 and shape[0] == 1:
            # Could be classes or scores
            # Distinguish by dtype: classes often float32 but integral values, scores float32 in [0,1]
            # We will assign later once both found; for now store candidates
            if 'classes' not in outs:
                outs['classes'] = data[0]
            else:
                outs['scores'] = data[0]
        elif len(shape) == 1 and shape[0] == 1:
            outs['count'] = int(np.squeeze(data).astype(np.int32))
    # Some models output classes and scores with ambiguous order; ensure they are set correctly
    # If both exist but we can't tell by type, assume the array with values > 1 is classes.
    if 'scores' not in outs or 'classes' not in outs:
        # Try to deduce among 2D arrays
        candidates = [interpreter.get_tensor(od['index'])[0] for od in output_details if len(interpreter.get_tensor(od['index']).shape) == 2]
        if len(candidates) >= 2:
            c1, c2 = candidates[0], candidates[1]
            # scores are typically between 0 and 1
            if np.mean(c1) <= 1.0 and np.max(c1) <= 1.0:
                outs['scores'] = c1
                outs['classes'] = c2
            else:
                outs['scores'] = c2
                outs['classes'] = c1
    # Fallbacks
    if 'count' not in outs:
        outs['count'] = len(outs.get('scores', []))
    return outs['boxes'], outs['classes'], outs['scores'], outs['count']


def denorm_box_to_xyxy(box, frame_w, frame_h):
    # box: [ymin, xmin, ymax, xmax] normalized [0,1]
    ymin, xmin, ymax, xmax = box
    x1 = int(max(0, min(frame_w - 1, xmin * frame_w)))
    y1 = int(max(0, min(frame_h - 1, ymin * frame_h)))
    x2 = int(max(0, min(frame_w - 1, xmax * frame_w)))
    y2 = int(max(0, min(frame_h - 1, ymax * frame_h)))
    return x1, y1, x2, y2


def draw_detections(frame, detections, labels, map_text):
    # detections: list of dicts with keys: bbox(x1,y1,x2,y2), class_id, score
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_id = det['class_id']
        score = det['score']
        label = labels.get(int(cls_id), str(int(cls_id)))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}: {score:.2f}"
        cv2.putText(frame, caption, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # Draw mAP text on top-left
    cv2.putText(frame, f"mAP@0.5: {map_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
    return frame


def iou_xyxy(box_a, box_b):
    # boxes: [x1,y1,x2,y2]
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_ap(recalls, precisions):
    # VOC-style interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Identify points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def compute_map(pred_by_frame, gt_by_frame, iou_threshold=0.5):
    # pred_by_frame: {frame_idx: [{'bbox':[x1,y1,x2,y2], 'class_id':int, 'score':float}, ...]}
    # gt_by_frame:   {frame_idx: [{'bbox':[x1,y1,x2,y2], 'class_id':int}, ...]}
    # Build class-wise lists
    classes = set()
    for _, preds in pred_by_frame.items():
        for p in preds:
            classes.add(int(p['class_id']))
    for _, gts in gt_by_frame.items():
        for g in gts:
            classes.add(int(g['class_id']))
    classes = sorted(list(classes))

    ap_list = []
    for cls in classes:
        # Gather all predictions of this class across frames
        cls_preds = []
        npos = 0
        gt_used_flags = {}
        for fidx, gts in gt_by_frame.items():
            # Count GT of this class for recall denominator
            gt_cls = [g for g in gts if int(g['class_id']) == cls]
            npos += len(gt_cls)
            gt_used_flags[fidx] = np.zeros(len(gt_cls), dtype=bool)

        # Collect predictions (frame_idx, score, bbox)
        for fidx, preds in pred_by_frame.items():
            for p in preds:
                if int(p['class_id']) == cls:
                    cls_preds.append((fidx, float(p['score']), p['bbox']))
        if len(cls_preds) == 0:
            if npos > 0:
                ap_list.append(0.0)
            continue

        # Sort predictions by descending score
        cls_preds.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, (fidx, score, pb) in enumerate(cls_preds):
            gts = gt_by_frame.get(fidx, [])
            gt_cls = [(j, g) for j, g in enumerate(gts) if int(g['class_id']) == cls]
            iou_max = 0.0
            jmax = -1
            for j, g in gt_cls:
                iou = iou_xyxy(pb, g['bbox'])
                if iou > iou_max:
                    iou_max = iou
                    jmax = j
            if iou_max >= iou_threshold and jmax != -1:
                # Need to map jmax within class-specific list to index in gt_used_flags for the frame
                # Build a mapping of class-specific to all gts indices for the frame
                # Simpler: rebuild class list to know index
                gt_cls_indices = [j for j, g in enumerate(gts) if int(g['class_id']) == cls]
                # Find the position of jmax among those
                cls_pos = gt_cls_indices.index(jmax)
                if not gt_used_flags.get(fidx, np.array([], dtype=bool)).size:
                    gt_used_flags[fidx] = np.zeros(len([g for g in gts if int(g['class_id']) == cls]), dtype=bool)
                if gt_used_flags[fidx].size <= cls_pos:
                    # Expand if needed (unlikely but safety)
                    new_flags = np.zeros(cls_pos + 1, dtype=bool)
                    new_flags[:gt_used_flags[fidx].size] = gt_used_flags[fidx]
                    gt_used_flags[fidx] = new_flags
                if not gt_used_flags[fidx][cls_pos]:
                    tp[i] = 1.0
                    gt_used_flags[fidx][cls_pos] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute precision-recall
        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        if npos == 0:
            # No ground truth of this class: ignore in mAP (common practice)
            continue
        recalls = tp_cum / float(npos)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        ap = compute_ap(recalls, precisions)
        ap_list.append(ap)

    if len(ap_list) == 0:
        return None
    return float(np.mean(ap_list))


def ensure_dir_for_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_ground_truth(gt_paths, labels):
    # Try multiple possible paths; return dict{frame_idx: [{'bbox':[x1,y1,x2,y2], 'class_id':int}, ...] }
    label_to_id = {name: idx for idx, name in labels.items()}
    for p in gt_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                gt_by_frame = {}

                def norm_obj(obj):
                    # Accept class as int or str
                    cls = obj.get("class", obj.get("label", None))
                    if isinstance(cls, str):
                        cls_id = label_to_id.get(cls, None)
                        if cls_id is None:
                            return None
                    else:
                        cls_id = int(cls)
                    bbox = obj.get("bbox", None)
                    if not bbox or len(bbox) != 4:
                        return None
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    return {"bbox": [x1, y1, x2, y2], "class_id": cls_id}

                if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
                    for item in data["frames"]:
                        fidx = int(item.get("frame_index", 0))
                        objs = []
                        for obj in item.get("objects", []):
                            nobj = norm_obj(obj)
                            if nobj:
                                objs.append(nobj)
                        gt_by_frame[fidx] = objs
                elif isinstance(data, dict):
                    # Mapping from frame index string to list of objects
                    for k, v in data.items():
                        try:
                            fidx = int(k)
                        except Exception:
                            continue
                        objs = []
                        for obj in v:
                            nobj = norm_obj(obj)
                            if nobj:
                                objs.append(nobj)
                        gt_by_frame[fidx] = objs
                elif isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        fidx = int(item.get("frame_index", 0))
                        objs = []
                        for obj in item.get("objects", []):
                            nobj = norm_obj(obj)
                            if nobj:
                                objs.append(nobj)
                        gt_by_frame[fidx] = objs
                else:
                    gt_by_frame = {}

                # Ensure ints, lists
                clean_gt = {int(k): list(v) for k, v in gt_by_frame.items()}
                return clean_gt, p
            except Exception:
                # If parse failed, try next
                continue
    return None, None


# =========================
# MAIN APPLICATION LOGIC
# =========================
def main():
    # 1. Setup: Load TFLite interpreter, allocate tensors; load labels; open video IO
    labels = load_labels(LABEL_PATH)
    interpreter, input_details, output_details = init_interpreter(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir_for_file(OUTPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    gt_by_frame, gt_path_used = load_ground_truth(POSSIBLE_GT_PATHS, labels)
    has_gt = gt_by_frame is not None

    pred_by_frame = {}

    frame_index = 0
    t0 = time.time()

    # 2-3. Process frames: preprocess, inference
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = prepare_input(frame, input_details)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes, classes, scores, count = get_outputs(interpreter, output_details)

        # Postprocess detections
        detections = []
        n = int(count) if count is not None else len(scores)
        for i in range(n):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(classes[i])
            x1, y1, x2, y2 = denorm_box_to_xyxy(boxes[i], width, height)
            # Skip invalid or tiny boxes
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "score": score
            })
        pred_by_frame[frame_index] = detections

        # 4. mAP calculation (running, if GT available up to current frame)
        map_text = "N/A"
        if has_gt:
            partial_pred = {k: v for k, v in pred_by_frame.items() if k <= frame_index}
            partial_gt = {k: v for k, v in gt_by_frame.items() if k <= frame_index}
            mAP_val = compute_map(partial_pred, partial_gt, iou_threshold=0.5)
            if mAP_val is not None:
                map_text = f"{mAP_val:.3f}"

        # Draw and write frame
        vis_frame = draw_detections(frame.copy(), detections, labels, map_text)
        out.write(vis_frame)
        frame_index += 1

    # Final mAP on the entire video (console)
    final_map_text = "N/A"
    if has_gt:
        final_mAP = compute_map(pred_by_frame, gt_by_frame, iou_threshold=0.5)
        if final_mAP is not None:
            final_map_text = f"{final_mAP:.4f}"

    cap.release()
    out.release()
    duration = time.time() - t0

    print("Processing complete.")
    print(f"Input video: {INPUT_PATH}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_index}")
    print(f"Total time: {duration:.2f}s  ({(frame_index / max(duration, 1e-6)):.2f} FPS)")
    if has_gt:
        print(f"Final mAP@0.5: {final_map_text} (GT file: {gt_path_used})")
    else:
        print("mAP not computed (no ground-truth file found).")


if __name__ == "__main__":
    main()