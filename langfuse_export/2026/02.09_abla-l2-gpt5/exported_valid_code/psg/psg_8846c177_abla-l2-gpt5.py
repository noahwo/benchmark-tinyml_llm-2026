import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =======================
# Configuration Parameters
# =======================
model_path = "models/ssd-mobilenet_v1/detect.tflite"
label_path = "models/ssd-mobilenet_v1/labelmap.txt"
input_path = "data/object_detection/sheeps.mp4"
output_path = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold = 0.5

# =======================
# Utility Functions
# =======================
def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if l:
                labels.append(l)
    return labels

def setup_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assume a single input tensor
    in_det = input_details[0]
    input_index = in_det['index']
    input_shape = in_det['shape']
    # Expected: [1, height, width, 3]
    if len(input_shape) != 4 or input_shape[-1] != 3:
        raise RuntimeError(f"Unexpected model input shape: {input_shape}")
    input_height, input_width = int(input_shape[1]), int(input_shape[2])
    input_dtype = in_det['dtype']

    return interpreter, input_index, input_height, input_width, input_dtype, output_details

def preprocess_frame(frame_bgr, input_width, input_height, input_dtype):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        input_data = resized.astype(np.float32) / 255.0
    else:
        input_data = resized.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def run_inference(interpreter, input_index, input_tensor, output_details):
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    # Try the common output order for TFLite detection models
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [N, 4] in ymin, xmin, ymax, xmax (normalized)
        classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)  # [N]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [N]
        num = int(interpreter.get_tensor(output_details[3]['index'])[0])  # scalar
    except Exception:
        # Fallback heuristic if order differs
        outs = [np.squeeze(interpreter.get_tensor(od['index'])) for od in output_details]
        boxes, classes, scores, num = None, None, None, None
        for o in outs:
            if o.ndim == 2 and o.shape[-1] == 4:
                boxes = o
            elif o.ndim == 1 and o.dtype in (np.float32, np.float64):
                # Could be scores or classes; pick later
                if scores is None:
                    scores = o
                else:
                    # whichever has more unique integers is likely classes after cast
                    cand = o
                    if len(np.unique(cand.astype(np.int32))) > len(np.unique(scores.astype(np.int32))):
                        classes = cand.astype(np.int32)
                    else:
                        scores = cand
            elif np.isscalar(o) or (o.ndim == 0):
                num = int(o)
        if boxes is None or classes is None or scores is None or num is None:
            raise RuntimeError("Unable to parse model outputs.")
    return boxes, classes, scores, num

def get_color_for_class(class_id):
    # Deterministic pseudo-color based on class_id
    # Map class_id into a BGR tuple
    r = (37 * (class_id + 1)) % 255
    g = (17 * (class_id + 7)) % 255
    b = (29 * (class_id + 13)) % 255
    return int(b), int(g), int(r)

def draw_detections(frame_bgr, boxes, classes, scores, num, labels, threshold, accum_scores_by_class):
    h, w = frame_bgr.shape[:2]
    drawn = 0
    num = min(num, boxes.shape[0], scores.shape[0], classes.shape[0])
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue
        cls_id = int(classes[i])
        ymin, xmin, ymax, xmax = boxes[i]
        # Convert normalized coords to pixel ints (clip to frame)
        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        color = get_color_for_class(cls_id)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label_text = labels[cls_id] if 0 <= cls_id < len(labels) else f"id:{cls_id}"
        text = f"{label_text}: {score:.2f}"
        # Draw background rectangle for text
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(frame_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + 4), color, thickness=-1)
        cv2.putText(frame_bgr, text, (x1 + 2, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Accumulate scores per class for proxy mAP
        if cls_id not in accum_scores_by_class:
            accum_scores_by_class[cls_id] = []
        accum_scores_by_class[cls_id].append(score)
        drawn += 1
    return drawn

def compute_proxy_map(accum_scores_by_class):
    # Proxy mAP: mean of average confidences per detected class.
    # This is NOT a true mAP (no ground truth available).
    if not accum_scores_by_class:
        return None
    per_class_means = []
    for cls_id, scores in accum_scores_by_class.items():
        if len(scores) > 0:
            per_class_means.append(float(np.mean(scores)))
    if not per_class_means:
        return None
    return float(np.mean(per_class_means))

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# =======================
# Main Pipeline
# =======================
def main():
    print("TFLite object detection - starting")
    print(f"Model: {model_path}")
    print(f"Labels: {label_path}")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print("Note: mAP shown is a proxy (mean confidence of detections per class) as no ground truth is provided.")

    labels = load_labels(label_path)
    # Common label maps include '???' at index 0; we keep it, since classes often are 1-based

    interpreter, input_index, in_h, in_w, in_dtype, output_details = setup_interpreter(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Unable to open input video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ensure_dir_for_file(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h), True)
    if not writer.isOpened():
        print(f"ERROR: Unable to create output video: {output_path}")
        cap.release()
        return

    accum_scores_by_class = {}
    processed = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame, in_w, in_h, in_dtype)
        boxes, classes, scores, num = run_inference(interpreter, input_index, input_tensor, output_details)

        # Draw detections and update accumulators
        draw_detections(frame, boxes, classes, scores, num, labels, confidence_threshold, accum_scores_by_class)

        # Compute proxy mAP so far and overlay on frame
        proxy_map = compute_proxy_map(accum_scores_by_class)
        map_text = f"mAP*: {proxy_map:.3f}" if proxy_map is not None else "mAP*: N/A"
        cv2.putText(frame, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 230), 2, cv2.LINE_AA)

        writer.write(frame)
        processed += 1
        if processed % 30 == 0:
            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0.0
            if total_frames > 0:
                print(f"Processed {processed}/{total_frames} frames at {speed:.2f} FPS")
            else:
                print(f"Processed {processed} frames at {speed:.2f} FPS")

    cap.release()
    writer.release()

    total_time = time.time() - t0
    final_map = compute_proxy_map(accum_scores_by_class)
    print("========== Summary ==========")
    print(f"Frames processed: {processed}")
    print(f"Total time: {total_time:.2f}s, Avg FPS: {processed / total_time if total_time > 0 else 0.0:.2f}")
    if final_map is None:
        print("Final mAP*: N/A (no detections or no ground truth available)")
    else:
        print(f"Final mAP*: {final_map:.4f}")
    print("* mAP* is a proxy metric (mean confidence of detections per class).")
    print(f"Saved annotated video to: {output_path}")

if __name__ == "__main__":
    main()