import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# -----------------------------------------------------------------------------
# CONFIGURATION PARAMETERS (provided)
# -----------------------------------------------------------------------------
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"  # corrected quotation
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
# input_description = "Read a single video file from the given input_path"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
# output_description = "Output the video file with rectangles drew on the detected objects, along with texts of labels and calculated mAP(mean average precision)"
confidence_threshold = 0.5

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load_labels(path):
    labels = {}
    try:
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Support "index label" or "label" per line
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1].strip()
                else:
                    # If no explicit indices, use line order
                    labels[idx] = line
    except Exception:
        # If any error occurs, fall back to empty labels (IDs will be used)
        labels = {}
    return labels

def make_interpreter(model_file):
    # Ensure Edge TPU delegate is used
    delegate_path = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"
    return Interpreter(model_path=model_file, experimental_delegates=[load_delegate(delegate_path)])

def preprocess(frame_bgr, input_details):
    # Prepare input according to model's expected size and dtype
    ih, iw = input_details[0]['shape'][1], input_details[0]['shape'][2]
    dtype = input_details[0]['dtype']
    resized = cv2.resize(frame_bgr, (iw, ih))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if dtype == np.uint8:
        input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
    else:
        # float models: normalize to [0,1]
        input_data = (np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0)
    return input_data

def dequantize_if_needed(detail, data):
    # Convert quantized output to float if necessary
    scale, zero_point = detail.get('quantization', (0.0, 0))
    if scale and scale > 0:
        return scale * (data.astype(np.float32) - zero_point)
    return data

def parse_detections(interpreter, output_details):
    boxes = None
    classes = None
    scores = None
    num = None

    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        data = dequantize_if_needed(od, data)
        name = od.get('name', '').lower()

        if boxes is None and (('box' in name) or (data.ndim == 3 and data.shape[-1] == 4)):
            boxes = data[0]
        elif scores is None and (('score' in name) or (data.ndim == 2 and data.shape[0] == 1)):
            # Many detection models have scores as [1, N]
            scores = data[0] if data.ndim == 2 else np.squeeze(data)
        elif classes is None and ('class' in name):
            raw = data[0] if data.ndim == 2 else np.squeeze(data)
            classes = raw.astype(np.int32)
        elif num is None and (('num' in name) or ('count' in name)):
            num = int(np.squeeze(data).astype(np.int32))

    # Fallbacks in case 'num' wasn't provided
    if num is None and scores is not None:
        num = int(scores.shape[0])
    if num is None and boxes is not None:
        num = int(boxes.shape[0])

    return boxes, classes, scores, num if num is not None else 0

def draw_label_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, fg=(255,255,255), bg=(0,0,0), thickness=1, pad=2):
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # Draw background rectangle
    cv2.rectangle(img, (x, y - h - baseline - pad), (x + w + pad*2, y + pad), bg, thickness=cv2.FILLED)
    # Put text above
    cv2.putText(img, text, (x + pad, y - baseline), font, font_scale, fg, thickness, cv2.LINE_AA)

def color_for_class(cid):
    # Generate a deterministic color for each class id
    np.random.seed(cid)
    c = np.random.randint(0, 255, size=3).tolist()
    return (int(c[0]), int(c[1]), int(c[2]))

def clip_box(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, min(w - 1, xmin))
    xmax = max(0, min(w - 1, xmax))
    ymin = max(0, min(h - 1, ymin))
    ymax = max(0, min(h - 1, ymax))
    return xmin, ymin, xmax, ymax

# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
def main():
    # Setup output directory
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labels
    labels = load_labels(label_path)

    # Initialize interpreter with EdgeTPU
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open video input
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open input video:", input_path)
        return

    # Prepare video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Error: Cannot open output video for writing:", output_path)
        cap.release()
        return

    # mAP placeholder (ground truth not provided)
    # The pipeline will overlay 'mAP: N/A' on the output frames.
    map_text = "mAP: N/A (no ground truth)"

    frame_count = 0
    t0_total = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Preprocess input
            input_tensor = preprocess(frame, input_details)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            infer_time_ms = (time.time() - t0) * 1000.0

            # Parse detections
            boxes, classes, scores, num = parse_detections(interpreter, output_details)

            # Draw detections
            if boxes is not None and scores is not None and classes is not None:
                for i in range(num):
                    score = float(scores[i])
                    if score < confidence_threshold:
                        continue
                    # TFLite models typically output normalized boxes [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = boxes[i]
                    if max(abs(ymin), abs(xmin), abs(ymax), abs(xmax)) <= 1.5:
                        # normalized
                        x0 = int(xmin * width)
                        y0 = int(ymin * height)
                        x1 = int(xmax * width)
                        y1 = int(ymax * height)
                    else:
                        # absolute
                        x0 = int(xmin)
                        y0 = int(ymin)
                        x1 = int(xmax)
                        y1 = int(ymax)

                    x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, width, height)
                    cid = int(classes[i])
                    label = labels.get(cid, f"id:{cid}")
                    color = color_for_class(cid)

                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    caption = f"{label} {score*100:.1f}%"
                    draw_label_with_bg(frame, caption, (x0, y0 if y0 > 20 else y0 + 20), bg=color, fg=(255,255,255))

            # Overlay mAP and performance info
            draw_label_with_bg(frame, map_text, (10, 25), bg=(0,0,0), fg=(255,255,255))
            draw_label_with_bg(frame, f"Infer: {infer_time_ms:.1f} ms", (10, 50), bg=(0,0,0), fg=(255,255,255))

            # Write frame
            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        total_time = time.time() - t0_total
        if frame_count > 0:
            print(f"Processed {frame_count} frames in {total_time:.2f}s ({frame_count/total_time:.2f} FPS).")
        else:
            print("No frames processed.")

if __name__ == "__main__":
    main()