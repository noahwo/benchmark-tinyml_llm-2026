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
# Utility Functions
# =========================
def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

def get_label_name(labels, class_id):
    # Handle both 0-based and 1-based label maps gracefully
    if 0 <= class_id < len(labels):
        return labels[class_id]
    if 0 <= (class_id - 1) < len(labels):
        return labels[class_id - 1]
    return str(class_id)

def make_color_for_class(cid):
    # Deterministic pseudo-random color per class id
    return (int((37 * cid) % 255), int((17 * cid) % 255), int((29 * cid) % 255))

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def preprocess_frame(frame_bgr, input_shape, input_dtype, quant_params):
    # input_shape: [1, H, W, C]
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    # Convert to RGB and resize
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    input_scale, input_zero_point = (0.0, 0)
    if isinstance(quant_params, (tuple, list)) and len(quant_params) == 2:
        input_scale, input_zero_point = quant_params

    if input_dtype == np.float32:
        input_data = (rgb.astype(np.float32) / 255.0).astype(np.float32)
    else:
        if input_scale and input_scale > 0:
            float_data = rgb.astype(np.float32) / 255.0
            quantized = np.round(float_data / input_scale + input_zero_point)
            if input_dtype == np.uint8:
                quantized = np.clip(quantized, 0, 255)
            elif input_dtype == np.int8:
                quantized = np.clip(quantized, -128, 127)
            input_data = quantized.astype(input_dtype)
        else:
            # Fallback: pass raw uint8 if no quant info available
            input_data = rgb.astype(input_dtype)

    # Add batch dimension if needed
    if input_data.ndim == 3:
        input_data = np.expand_dims(input_data, axis=0)
    return input_data

def parse_outputs(interpreter):
    # Attempt to extract outputs using names first, else fallback by shape/dtype heuristics
    output_details = interpreter.get_output_details()
    boxes = classes = scores = num = None

    # Try name-based mapping (typical for TFLite SSD models)
    name_map = {}
    for od in output_details:
        name = od.get('name', b'')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='ignore')
        name_map[name] = od

    def get_tensor_by_detail(od):
        return interpreter.get_tensor(od['index'])

    # Heuristic 1: name-based
    if any('TFLite_Detection_PostProcess' in k for k in name_map.keys()):
        for k, od in name_map.items():
            data = get_tensor_by_detail(od)
            if k.endswith(':0') or k.endswith('PostProcess') or data.ndim == 3 and data.shape[-1] == 4:
                boxes = data
            elif k.endswith(':1'):
                classes = data
            elif k.endswith(':2'):
                scores = data
            elif k.endswith(':3'):
                num = data

    # Heuristic 2: fallback by shapes/dtypes if any missing
    if boxes is None or classes is None or scores is None:
        for od in output_details:
            data = get_tensor_by_detail(od)
            if data.ndim == 3 and data.shape[-1] == 4 and boxes is None:
                boxes = data
            elif data.ndim >= 2 and np.issubdtype(data.dtype, np.floating) and scores is None and (data.shape[-1] != 4):
                scores = data
            elif data.ndim >= 2 and not np.issubdtype(data.dtype, np.floating) and classes is None:
                classes = data
            elif data.size == 1 and num is None:
                num = data

    # Squeeze/reshape to standard shapes
    if boxes is not None:
        boxes = np.squeeze(boxes)
    if classes is not None:
        classes = np.squeeze(classes).astype(int)
    if scores is not None:
        scores = np.squeeze(scores).astype(np.float32)
    if num is not None:
        try:
            num_val = int(np.squeeze(num).astype(int))
        except Exception:
            num_val = int(np.squeeze(num))
    else:
        num_val = None

    # Truncate to num_detections if provided
    if num_val is not None and boxes is not None and scores is not None and classes is not None:
        boxes = boxes[:num_val]
        classes = classes[:num_val]
        scores = scores[:num_val]

    return boxes, classes, scores

def draw_detections(frame_bgr, detections, labels, conf_threshold):
    h, w = frame_bgr.shape[:2]
    boxes, classes, scores = detections
    drawn = 0
    per_class_counts = {}

    if boxes is None or classes is None or scores is None:
        return frame_bgr, drawn, per_class_counts

    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf_threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        # Boxes are typically normalized [0,1]; detect if not by checking ranges
        if 0.0 <= y_min <= 1.0 and 0.0 <= y_max <= 1.0 and 0.0 <= x_min <= 1.0 and 0.0 <= x_max <= 1.0:
            y1 = int(max(0, min(h - 1, y_min * h)))
            x1 = int(max(0, min(w - 1, x_min * w)))
            y2 = int(max(0, min(h - 1, y_max * h)))
            x2 = int(max(0, min(w - 1, x_max * w)))
        else:
            y1, x1, y2, x2 = int(y_min), int(x_min), int(y_max), int(x_max)
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))

        cid = int(classes[i]) if i < len(classes) else -1
        label = get_label_name(labels, cid)
        color = make_color_for_class(cid if cid >= 0 else 0)

        # Draw rectangle
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # Draw label background and text
        caption = f"{label}: {score*100:.1f}%"
        (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - 4)
        cv2.rectangle(frame_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + 4), color, thickness=-1)
        cv2.putText(frame_bgr, caption, (x1 + 2, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Count detections per class for proxy mAP
        if cid not in per_class_counts:
            per_class_counts[cid] = 0
        per_class_counts[cid] += 1

        drawn += 1

    return frame_bgr, drawn, per_class_counts

# Proxy mAP aggregator using detection counts (since no ground-truth is provided).
# For each class and frame: precision = 1 / n_detections_for_class_in_frame (if > 0).
# AP_class = mean over frames of that precision; mAP = mean over classes.
class ProxyMAP:
    def __init__(self):
        self.sum_inv_counts = {}   # class_id -> sum of (1 / n_dets_in_frame)
        self.frames_with_det = {}  # class_id -> number of frames where class appeared

    def update(self, per_class_counts):
        for cid, n in per_class_counts.items():
            if n > 0:
                self.sum_inv_counts[cid] = self.sum_inv_counts.get(cid, 0.0) + (1.0 / float(n))
                self.frames_with_det[cid] = self.frames_with_det.get(cid, 0) + 1

    def value(self):
        ap_values = []
        for cid, denom in self.frames_with_det.items():
            if denom > 0:
                ap = self.sum_inv_counts.get(cid, 0.0) / float(denom)
                ap_values.append(ap)
        if not ap_values:
            return None
        return float(np.mean(ap_values))

# =========================
# Main Application
# =========================
def main():
    # Load labels
    labels = load_labels(label_path)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # [1, H, W, C]
    input_dtype = input_details[0]['dtype']
    input_quant_params = input_details[0].get('quantization', (0.0, 0))

    # Video IO setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # Fallback FPS if unavailable
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    # Proxy mAP aggregator
    proxy_map = ProxyMAP()

    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Preprocess
            input_data = preprocess_frame(frame_bgr, input_shape, input_dtype, input_quant_params)

            # Inference
            t0 = time.time()
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            inf_time_ms = (time.time() - t0) * 1000.0

            # Parse detections
            boxes, classes, scores = parse_outputs(interpreter)

            # Draw detections
            frame_bgr, drawn_count, per_class_counts = draw_detections(
                frame_bgr, (boxes, classes, scores), labels, confidence_threshold
            )

            # Update proxy mAP
            proxy_map.update(per_class_counts)
            current_map = proxy_map.value()

            # Overlay runtime info and mAP
            overlay_y = 20
            info_color = (0, 0, 0)
            bg_color = (255, 255, 255)

            def put_overlay(text, y):
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_bgr, (5, y - th - 2), (5 + tw + 4, y + 4), bg_color, -1)
                cv2.putText(frame_bgr, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1, cv2.LINE_AA)
                return y + th + 10

            overlay_y = put_overlay(f"Detections: {drawn_count} | Inference: {inf_time_ms:.1f} ms", overlay_y)
            if current_map is None:
                overlay_y = put_overlay("mAP (proxy): N/A", overlay_y)
            else:
                overlay_y = put_overlay(f"mAP (proxy): {current_map:.3f}", overlay_y)

            # Write frame
            out.write(frame_bgr)
            frame_idx += 1

    finally:
        cap.release()
        out.release()

    total_time = time.time() - t_start
    print(f"Processed {frame_idx} frames in {total_time:.2f}s "
          f"({(frame_idx/total_time) if total_time>0 else 0:.2f} FPS).")
    final_map = proxy_map.value()
    if final_map is None:
        print("Final mAP (proxy): N/A")
    else:
        print(f"Final mAP (proxy): {final_map:.4f}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()