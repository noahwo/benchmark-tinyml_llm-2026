#!/usr/bin/env python3
"""
Object Detection via a video file on Raspberry Pi 4B using a TFLite SSD model.

Phases implemented per guideline:
- Phase 1: Setup (imports, parameters, labels, interpreter, model details)
- Phase 2: Input Acquisition & Preprocessing Loop (video file reading, resizing, dtype handling)
- Phase 3: Inference
- Phase 4: Output Interpretation & Handling Loop
  - 4.2 Interpret Results (labels mapping, boxes/classes/scores extraction)
  - 4.3 Post-processing (thresholding, scaling, clipping)
  - 4.4 Handle Output (draw rectangles and labels, write output video, overlay proxy mAP)
- Phase 5: Cleanup

Note on mAP:
- Since no ground-truth annotations are provided, a true mAP cannot be computed.
- This script computes a practical proxy metric: per-class AP_proxy = mean confidence of detections for that class,
  and mAP_proxy = mean(AP_proxy) across classes that had at least one detection. It is reported on-console and overlaid on the video.
"""

import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: Interpreter from ai_edge_litert
from ai_edge_litert.interpreter import Interpreter

def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name != '':
                labels.append(name)
    return labels

def ensure_dir_for_file(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def iou(boxA, boxB):
    # boxes in (x1, y1, x2, y2) format
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    return [x1, y1, x2, y2]

def make_color_palette(num_classes):
    # Fixed palette for reproducibility; HSV-based
    colors = []
    for i in range(num_classes):
        hue = int(179 * (i / max(1, num_classes)))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0,0,:].tolist()
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    if num_classes == 0:
        colors.append((0, 255, 0))
    return colors

def main():
    # 1.2 Paths/Parameters
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = 0.5  # as provided

    # 1.3 Load Labels (conditional/relevant)
    labels = load_labels(label_path)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assume single input tensor
    input_index = input_details[0]['index']
    in_dtype = input_details[0]['dtype']
    in_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]
    # Some interpreters return tensor index mapping; capture outputs generically
    output_indices = [od['index'] for od in output_details]

    # Define input size from model
    if len(in_shape) == 4:
        batch, in_h, in_w, in_c = in_shape
    else:
        raise RuntimeError("Unexpected input tensor shape: {}".format(in_shape))

    # Floating model?
    floating_model = (in_dtype == np.float32)

    # Prepare color palette for labels
    colors = make_color_palette(len(labels))

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data: open the single video file specified by input_path
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input video not found: {}".format(input_path))

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video: {}".format(input_path))

    # Prepare output writer
    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0  # fallback
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open output video for writing: {}".format(output_path))

    # Storage for proxy mAP computation: per-class list of confidences
    class_confidences = {}  # class_id -> list of conf scores

    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # 2.2 Preprocess Data
        # Convert BGR to RGB for typical TFLite models
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

        # 2.3 Quantization Handling
        if floating_model:
            # Normalize to [-1, 1] as per guideline
            input_data = (np.float32(resized) - 127.5) / 127.5
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        else:
            # uint8 quantized models expect [0,255]
            input_data = np.expand_dims(resized, axis=0).astype(in_dtype)

        # Phase 3: Inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling Loop
        # 4.1 Get Output Tensors
        # SSD MobileNet typically returns: boxes, classes, scores, num_detections
        # We'll identify them by shapes/dtypes
        raw_outputs = [interpreter.get_tensor(idx) for idx in output_indices]

        boxes = None
        classes_out = None
        scores = None
        num = None

        # Heuristic mapping
        for arr in raw_outputs:
            arr_np = np.array(arr)
            if arr_np.ndim == 3 and arr_np.shape[-1] == 4:
                boxes = arr_np  # shape [1, N, 4]
            elif arr_np.ndim == 2 and arr_np.shape[0] == 1 and arr_np.shape[1] > 1 and arr_np.dtype in [np.float32, np.float64]:
                # could be scores or classes depending on dtype
                # scores are float, classes often float in TFLite SSD
                # We'll check value ranges later
                # We'll assign after collecting candidates
                pass
            elif arr_np.ndim == 2 and arr_np.shape[0] == 1 and arr_np.shape[1] == 1:
                num = int(np.squeeze(arr_np).astype(np.int32))

        # Extract classes and scores by inspecting remaining arrays
        cand = [np.array(a) for a in raw_outputs]
        # Remove boxes and num from candidates to avoid confusion
        cand_filt = []
        for a in cand:
            if boxes is not None and a.shape == boxes.shape and a.ndim == 3:
                continue
            if num is not None and a.size == 1:
                continue
            cand_filt.append(a)

        # Among candidates, one is classes (float or int), one is scores (float)
        classes_arr = None
        scores_arr = None
        for a in cand_filt:
            a_squeezed = np.squeeze(a)
            if a_squeezed.ndim == 1:
                # If all values are small integers or near-integers, it's likely classes
                if np.all(np.abs(a_squeezed - np.round(a_squeezed)) < 1e-3):
                    classes_arr = a
                else:
                    scores_arr = a
            elif a_squeezed.ndim == 2 and a_squeezed.shape[0] == 1:
                # Typical shape [1, N]
                vals = a_squeezed[0]
                if np.all(np.abs(vals - np.round(vals)) < 1e-3):
                    classes_arr = a
                else:
                    scores_arr = a

        # Fallback if mapping ambiguous: pick by dtype
        if classes_arr is None or scores_arr is None:
            for a in cand_filt:
                if a.dtype in (np.uint8, np.int32, np.int64):
                    classes_arr = a
                elif a.dtype in (np.float32, np.float64):
                    scores_arr = a

        # Squeeze to 1D arrays
        if boxes is None:
            raise RuntimeError("Could not find detection boxes in model outputs.")
        boxes = np.squeeze(boxes)  # [N,4] y_min, x_min, y_max, x_max in normalized coords

        if classes_arr is None or scores_arr is None:
            raise RuntimeError("Could not map classes/scores from model outputs.")

        classes_out = np.squeeze(classes_arr)  # [N]
        scores = np.squeeze(scores_arr)        # [N]
        # Some models return float class indices; cast to int
        classes_out = classes_out.astype(np.int32)

        # 4.2 Interpret Results: map indices to names, prepare detections
        # 4.3 Post-processing: confidence thresholding and coordinate scaling/clipping
        detections = []
        H, W = frame_bgr.shape[:2]
        for i in range(boxes.shape[0]):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            cls_id = int(classes_out[i])
            # Safety check for labels range
            label = labels[cls_id] if 0 <= cls_id < len(labels) else f"id_{cls_id}"
            y_min, x_min, y_max, x_max = boxes[i].tolist()
            # Scale to absolute pixel coordinates
            x1 = x_min * W
            y1 = y_min * H
            x2 = x_max * W
            y2 = y_max * H
            x1, y1, x2, y2 = clip_box_xyxy([x1, y1, x2, y2], W, H)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': score,
                'class_id': cls_id,
                'label': label
            })

        # Accumulate confidences for proxy mAP
        for det in detections:
            cid = det['class_id']
            class_confidences.setdefault(cid, []).append(det['score'])

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            color = colors[cls_id % len(colors)] if len(colors) > 0 else (0, 255, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label_text = "{}: {:.2f}".format(det['label'], det['score'])
            # Background label box
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - th - baseline), (x1 + tw, y1), color, thickness=-1)
            cv2.putText(frame_bgr, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Compute running proxy mAP
        ap_vals = []
        for cid, conf_list in class_confidences.items():
            if len(conf_list) > 0:
                ap_vals.append(float(np.mean(conf_list)))
        mAP_proxy = float(np.mean(ap_vals)) if len(ap_vals) > 0 else 0.0

        # Overlay mAP proxy and FPS
        elapsed = time.time() - t_start
        fps_infer = frame_count / elapsed if elapsed > 0 else 0.0
        overlay_text = "mAP(proxy): {:.3f} | FPS: {:.1f}".format(mAP_proxy, fps_infer)
        cv2.putText(frame_bgr, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)

        # 4.4 Handle Output: write the annotated frame to output video
        writer.write(frame_bgr)

        # 4.5 Loop Continuation: continue until video ends (handled by while loop)

    # After processing all frames, finalize mAP proxy and report
    final_ap_per_class = {}
    for cid, conf_list in class_confidences.items():
        if len(conf_list) > 0:
            final_ap_per_class[cid] = float(np.mean(conf_list))
    final_mAP_proxy = float(np.mean(list(final_ap_per_class.values()))) if len(final_ap_per_class) > 0 else 0.0

    print("==== Inference Summary ====")
    print("Processed frames:", frame_count)
    print("Classes detected:", len(final_ap_per_class))
    # Print top classes by proxy AP
    if len(final_ap_per_class) > 0:
        sorted_items = sorted(final_ap_per_class.items(), key=lambda kv: kv[1], reverse=True)
        for cid, ap in sorted_items[:10]:
            cname = labels[cid] if 0 <= cid < len(labels) else f"id_{cid}"
            print(f"Class: {cname:20s} AP_proxy: {ap:.4f}")
    print("mAP(proxy): {:.4f}".format(final_mAP_proxy))
    print("Output video written to:", output_path)

    # Phase 5: Cleanup
    cap.release()
    writer.release()

if __name__ == "__main__":
    main()