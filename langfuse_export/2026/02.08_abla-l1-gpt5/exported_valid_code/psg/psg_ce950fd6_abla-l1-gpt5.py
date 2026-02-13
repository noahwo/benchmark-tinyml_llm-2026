import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# CONFIGURATION PARAMETERS
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                labels.append(name)
    return labels

def main():
    # 1) SETUP
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    labels = load_labels(LABEL_PATH)

    # Determine label offset (common labelmap starts with '???' or 'background')
    label_offset = 1 if (len(labels) > 0 and labels[0].strip().lower() in ("???", "background")) else 0

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assume single input
    input_detail = input_details[0]
    input_shape = input_detail["shape"]
    input_height, input_width = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_detail["dtype"]

    # Video IO initialization
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {INPUT_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"Error: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # 2) PREPROCESSING, 3) INFERENCE, 4) OUTPUT HANDLING
    frame_index = 0
    t_start = time.time()
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_index += 1

        # Preprocessing: BGR -> RGB, resize, normalize/cast
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

        if input_dtype == np.float32:
            input_tensor = (resized.astype(np.float32) / 255.0)
        else:
            input_tensor = resized.astype(input_dtype)

        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Inference
        interpreter.set_tensor(input_detail["index"], input_tensor)
        interpreter.invoke()

        # Retrieve outputs robustly
        det_boxes = None
        det_classes = None
        det_scores = None
        det_num = None

        for od in output_details:
            out = interpreter.get_tensor(od["index"])
            # Typical outputs:
            # boxes: [1, num, 4]
            # classes: [1, num]
            # scores: [1, num]
            # num_detections: [1]
            if out.ndim == 3 and out.shape[-1] == 4:
                det_boxes = out[0]
            elif out.ndim == 2 and out.shape[0] == 1:
                arr = out[0]
                # Heuristic: scores are in [0,1], classes are integer-like >= 0
                if np.max(arr) <= 1.0 + 1e-6:
                    det_scores = arr
                else:
                    det_classes = arr
            elif out.size == 1:
                det_num = int(round(float(out.reshape(-1)[0])))

        # Fallback shapes if any missing
        if det_boxes is None:
            # Try find any output that can be boxes
            for od in output_details:
                out = interpreter.get_tensor(od["index"])
                if out.ndim == 2 and out.shape[-1] == 4:
                    det_boxes = out
                    break

        if det_scores is None or det_boxes is None or det_classes is None:
            # If outputs are not as expected, just write the original frame
            writer.write(frame_bgr)
            continue

        num = min(len(det_scores), len(det_boxes), len(det_classes))
        if det_num is not None:
            num = min(num, det_num)

        # Draw detections
        for i in range(num):
            score = float(det_scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(det_classes[i])
            label_index = cls_id + label_offset
            if 0 <= label_index < len(labels):
                label_name = labels[label_index]
            elif 0 <= cls_id < len(labels):
                label_name = labels[cls_id]
            else:
                label_name = f"class_{cls_id}"

            # Boxes are [ymin, xmin, ymax, xmax] in normalized coordinates
            y_min, x_min, y_max, x_max = det_boxes[i]
            x0 = max(0, min(frame_w - 1, int(x_min * frame_w)))
            y0 = max(0, min(frame_h - 1, int(y_min * frame_h)))
            x1 = max(0, min(frame_w - 1, int(x_max * frame_w)))
            y1 = max(0, min(frame_h - 1, int(y_max * frame_h)))

            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            caption = f"{label_name}: {score:.2f}"
            # Text background
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(0, y0 - th - baseline)
            cv2.rectangle(frame_bgr, (x0, ty), (x0 + tw, ty + th + baseline), (0, 255, 0), -1)
            cv2.putText(frame_bgr, caption, (x0, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(frame_bgr)

    t_end = time.time()
    cap.release()
    writer.release()
    total_time = t_end - t_start
    if frame_index > 0:
        print(f"Processed {frame_index} frames in {total_time:.2f}s ({frame_index / max(total_time, 1e-6):.2f} FPS).")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()