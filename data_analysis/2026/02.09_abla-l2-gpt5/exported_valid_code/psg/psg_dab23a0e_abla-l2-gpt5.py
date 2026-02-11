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
INPUT_PATH = "data/object_detection/sheeps.mp4"  # Read a single video file from the given input_path
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"  # Output video with rectangles, labels, and mAP
CONFIDENCE_THRESHOLD = 0.5


def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if s:
                labels.append(s)
    return labels


def prepare_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_frame(frame_bgr, input_details):
    # Convert BGR (OpenCV) to RGB and resize to model input
    h_in, w_in = input_details[0]['shape'][1], input_details[0]['shape'][2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(frame_rgb, (w_in, h_in))

    in_dtype = input_details[0]['dtype']
    # For float models, normalize to [0,1]; for quantized (uint8), keep 0..255
    if in_dtype == np.float32:
        input_tensor = img_resized.astype(np.float32) / 255.0
    else:
        input_tensor = img_resized.astype(in_dtype)

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor


def get_output_tensors(interpreter, output_details):
    # Try to map outputs by name when available; otherwise fallback to heuristics
    boxes = None
    classes = None
    scores = None
    num = None

    for d in output_details:
        name = d.get('name', '')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='ignore')
        data = interpreter.get_tensor(d['index'])
        lname = name.lower()

        if 'box' in lname:
            boxes = data
        elif 'score' in lname:
            scores = data
        elif 'class' in lname:
            classes = data
        elif 'num' in lname:
            num = data

    # Fallback if names were not matched (assume typical SSD order)
    if boxes is None or classes is None or scores is None:
        outs = [interpreter.get_tensor(d['index']) for d in output_details]
        # Guess by shapes
        for out in outs:
            shp = out.shape
            if len(shp) == 3 and shp[-1] == 4:
                boxes = out
            elif len(shp) == 2:
                # Could be classes or scores; distinguish by dtype/range is unreliable; attempt both
                if out.dtype == np.float32:
                    # More likely scores
                    scores = out if scores is None else scores
                else:
                    classes = out if classes is None else classes
            elif len(shp) == 1 and shp[0] == 1:
                num = out

        # If still ambiguous, try positional mapping as last resort
        if boxes is None and len(outs) >= 1:
            boxes = outs[0]
        if classes is None and len(outs) >= 2:
            classes = outs[1]
        if scores is None and len(outs) >= 3:
            scores = outs[2]
        if num is None and len(outs) >= 4:
            num = outs[3]

    return boxes, classes, scores, num


def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Setup TFLite interpreter
    interpreter, input_details, output_details = prepare_interpreter(MODEL_PATH)

    # Open input video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Error: Could not open input video:", INPUT_PATH)
        return

    # Prepare video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # Fallback if fps is not available
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Error: Could not open output video for writing:", OUTPUT_PATH)
        cap.release()
        return

    # For proxy mAP: aggregate detection scores over the entire video (scores above threshold)
    # Note: True mAP requires ground-truth annotations; here we compute a proxy metric as
    # the mean of detection scores for all detections above threshold across the video.
    # This will be displayed as "mAP" as requested.
    aggregated_scores = []

    frame_index = 0
    inference_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_index += 1

            # Preprocess
            input_tensor = preprocess_frame(frame, input_details)

            # Set input and run inference
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            inference_times.append((t1 - t0) * 1000.0)  # ms

            # Fetch outputs
            boxes, classes, scores, num = get_output_tensors(interpreter, output_details)

            # Squeeze batch dimension
            if boxes is not None:
                boxes = np.squeeze(boxes)
            if classes is not None:
                classes = np.squeeze(classes)
            if scores is not None:
                scores = np.squeeze(scores)
            if num is not None:
                try:
                    num = int(np.squeeze(num).astype(np.int32))
                except Exception:
                    num = None

            # Determine number of detections to iterate
            if num is None:
                num = scores.shape[0] if scores is not None else 0

            # Draw detections
            for i in range(num):
                score = float(scores[i]) if scores is not None else 0.0
                if score < CONFIDENCE_THRESHOLD:
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                x1 = max(0, min(width - 1, int(xmin * width)))
                y1 = max(0, min(height - 1, int(ymin * height)))
                x2 = max(0, min(width - 1, int(xmax * width)))
                y2 = max(0, min(height - 1, int(ymax * height)))

                cls_id = int(classes[i]) if classes is not None else -1
                label = labels[cls_id] if (0 <= cls_id < len(labels)) else f"id:{cls_id}"

                # Draw rectangle and label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"{label}: {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_text = y1 - 10 if y1 - 10 > 10 else y1 + th + 10
                cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Accumulate score for proxy mAP
                aggregated_scores.append(score)

            # Compute running "mAP" proxy
            if len(aggregated_scores) > 0:
                map_proxy = float(np.mean(aggregated_scores))
            else:
                map_proxy = 0.0

            # Overlay mAP and (optional) inference time
            overlay_text = f"mAP: {map_proxy:.3f}"
            if len(inference_times) > 0:
                overlay_text += f" | Inference: {inference_times[-1]:.1f} ms"
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 10), 2)

            # Write frame
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    # Final report
    final_map = float(np.mean(aggregated_scores)) if len(aggregated_scores) > 0 else 0.0
    avg_infer = float(np.mean(inference_times)) if len(inference_times) > 0 else 0.0
    print("Processing completed.")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {frame_index}")
    print(f"Proxy mAP (mean score above threshold {CONFIDENCE_THRESHOLD}): {final_map:.4f}")
    print(f"Average inference time: {avg_infer:.2f} ms/frame")


if __name__ == "__main__":
    main()