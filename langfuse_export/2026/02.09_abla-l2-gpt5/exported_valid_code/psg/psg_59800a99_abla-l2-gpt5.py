import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"         # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Output the video file with rectangles, labels, and mAP
CONF_THRESHOLD = 0.5


def load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                # Handle potential "index name" format or plain name
                # If the line has an index and a name, split and keep the name part.
                parts = name.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1])
                else:
                    labels.append(name)
    return labels


def get_input_size_and_dtype(interpreter):
    input_details = interpreter.get_input_details()
    if not input_details:
        raise RuntimeError("No input details found in the interpreter.")
    input_shape = input_details[0]["shape"]
    # Expect shape [1, height, width, channels]
    if len(input_shape) != 4:
        raise RuntimeError(f"Unexpected input tensor shape: {input_shape}")
    height, width = int(input_shape[1]), int(input_shape[2])
    dtype = input_details[0]["dtype"]
    return width, height, dtype


def preprocess_frame(frame_bgr, input_w, input_h, input_dtype):
    # Convert BGR to RGB for most TFLite detection models
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    if input_dtype == np.float32:
        tensor = resized.astype(np.float32) / 255.0
    else:
        # For uint8/other types, pass as is (TFLite quantization parameters will be used internally)
        tensor = resized.astype(input_dtype)
    tensor = np.expand_dims(tensor, axis=0)  # [1, H, W, C]
    return tensor


def run_inference(interpreter, input_tensor):
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    # Typical SSD MobileNet v1 TFLite outputs:
    # 0: detection_boxes [1, num, 4] (ymin, xmin, ymax, xmax) normalized
    # 1: detection_classes [1, num]
    # 2: detection_scores [1, num]
    # 3: num_detections [1]
    # However, order may vary; we'll infer by tensor shapes.
    boxes = classes = scores = num = None
    for od in output_details:
        out = interpreter.get_tensor(od["index"])
        if out.ndim == 3 and out.shape[-1] == 4:
            boxes = out[0]
        elif out.ndim == 2:
            # Could be classes or scores depending on dtype
            if out.dtype in (np.float32, np.float64):
                # assume scores
                scores = out[0].astype(np.float32)
            else:
                classes = out[0].astype(np.int32)
        elif out.size == 1:
            num = int(np.squeeze(out).astype(np.int32))

    # Fallback if num_detections not present
    if num is None and scores is not None:
        num = len(scores)

    # Trim arrays to num
    if boxes is not None:
        boxes = boxes[:num]
    if classes is not None:
        classes = classes[:num]
    if scores is not None:
        scores = scores[:num]

    return boxes, classes, scores, num


def label_for_class(class_id, labels):
    # Try direct index
    if 0 <= class_id < len(labels):
        return labels[class_id]
    # Many TFLite SSD models are 1-based in label files (first entry '???')
    if 0 <= (class_id + 1) < len(labels):
        return labels[class_id + 1]
    return f"id:{class_id}"


def draw_detections(frame_bgr, boxes, classes, scores, labels, conf_thr):
    h, w = frame_bgr.shape[:2]
    drawn = 0
    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf_thr:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        # Convert from normalized [0,1] to absolute pixels
        x1 = int(max(0, xmin) * w)
        y1 = int(max(0, ymin) * h)
        x2 = int(min(1.0, xmax) * w)
        y2 = int(min(1.0, ymax) * h)

        cls_id = int(classes[i]) if classes is not None else -1
        cls_name = label_for_class(cls_id, labels) if labels else f"id:{cls_id}"

        color = (0, 200, 0)  # Green
        thickness = 2
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{cls_name}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Draw filled rectangle for text background
        y_text = max(0, y1 - th - 4)
        cv2.rectangle(frame_bgr, (x1, y_text), (x1 + tw + 4, y_text + th + baseline + 4), color, -1)
        cv2.putText(
            frame_bgr, label_text, (x1 + 2, y_text + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        drawn += 1
    return drawn


def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_w, input_h, input_dtype = get_input_size_and_dtype(interpreter)

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_PATH}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {OUTPUT_PATH}")

    # For runtime stats and proxy mAP calculation (no ground-truth available)
    proxy_scores_all = []  # Collect all detection scores above threshold
    frame_index = 0
    t0_total = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            # Preprocess
            input_tensor = preprocess_frame(frame, input_w, input_h, input_dtype)

            # Inference
            t0 = time.time()
            boxes, classes, scores, num = run_inference(interpreter, input_tensor)
            infer_time_ms = (time.time() - t0) * 1000.0

            # Draw detections
            if boxes is None or scores is None:
                # No outputs; write frame as-is
                display = frame.copy()
                map_text = "mAP (proxy): N/A"
                cv2.putText(display, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                writer.write(display)
                continue

            display = frame.copy()
            # Accumulate scores above threshold for a proxy "mAP" in absence of ground truth
            above_thr = [float(s) for s in scores if s >= CONF_THRESHOLD]
            proxy_scores_all.extend(above_thr)

            _ = draw_detections(display, boxes, classes, scores, labels, CONF_THRESHOLD)

            # Compute running proxy mAP (mean of detection scores above threshold)
            if proxy_scores_all:
                map_proxy = float(np.mean(proxy_scores_all))
                map_text = f"mAP (proxy): {map_proxy:.3f}"
            else:
                map_text = "mAP (proxy): N/A"

            # Overlay info: proxy mAP and inference time
            cv2.putText(display, map_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 170, 50), 2, cv2.LINE_AA)
            perf_text = f"Infer: {infer_time_ms:.1f} ms"
            cv2.putText(display, perf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 220), 2, cv2.LINE_AA)

            writer.write(display)

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - t0_total
    # Final proxy mAP summary in console
    if proxy_scores_all:
        final_map_proxy = float(np.mean(proxy_scores_all))
        print(f"Finished. Frames: {frame_index}, Time: {total_time:.2f}s, Proxy mAP: {final_map_proxy:.4f}")
    else:
        print(f"Finished. Frames: {frame_index}, Time: {total_time:.2f}s, Proxy mAP: N/A (no detections above threshold)")

    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()