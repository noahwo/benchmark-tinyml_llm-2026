import os
import time
import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# Configuration parameters
MODEL_PATH = "models/ssd-mobilenet_v1/detect.tflite"
LABEL_PATH = "models/ssd-mobilenet_v1/labelmap.txt"
INPUT_PATH = "data/object_detection/sheeps.mp4"
OUTPUT_PATH = "results/object_detection/test_results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD = 0.5

def load_labels(path):
    labels = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception:
        pass
    return labels

def get_label_text(labels, class_id):
    # Try common mappings (some label files are 1-based, some 0-based, some have '???' first)
    text = str(class_id)
    if labels:
        if 0 <= class_id < len(labels):
            text = labels[class_id]
        elif 1 <= class_id <= len(labels):
            text = labels[class_id - 1]
    return text

def make_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame, input_size, input_dtype):
    # Resize and convert BGR to RGB
    h_in, w_in = input_size
    resized = cv2.resize(frame, (w_in, h_in))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        inp = (rgb.astype(np.float32) / 255.0)
    elif input_dtype == np.uint8:
        inp = rgb.astype(np.uint8)
    else:
        # Fallback: cast to expected dtype without scaling
        inp = rgb.astype(input_dtype)
    # Add batch dimension
    inp = np.expand_dims(inp, axis=0)
    return inp

def find_output_tensors(interpreter):
    # Attempts to identify boxes, classes, scores, and num_detections tensors
    output_details = interpreter.get_output_details()
    boxes = classes = scores = num = None
    arrays = [interpreter.get_tensor(d['index']) for d in output_details]

    # Heuristic detection by shape/content
    for arr in arrays:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr[0]
    for arr in arrays:
        if arr.ndim == 2 and arr.shape[0] == 1:
            # Scores typically in [0,1]
            if arr.dtype.kind == 'f' and np.max(arr) <= 1.0 and np.min(arr) >= 0.0:
                scores = arr[0]
            else:
                classes = arr[0].astype(np.int32)
    for arr in arrays:
        if arr.ndim == 1 and arr.size == 1:
            try:
                num = int(np.round(arr[0]))
            except Exception:
                num = None

    # Fallback to standard TF Lite SSD ordering if any missing
    if boxes is None or classes is None or scores is None:
        # Standard ordering: [boxes, classes, scores, num]
        try:
            boxes_fb = interpreter.get_tensor(output_details[0]['index'])[0]
            classes_fb = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
            scores_fb = interpreter.get_tensor(output_details[2]['index'])[0]
            num_fb = int(np.round(interpreter.get_tensor(output_details[3]['index'])[0])) if len(output_details) > 3 else None
            boxes = boxes if boxes is not None else boxes_fb
            classes = classes if classes is not None else classes_fb
            scores = scores if scores is not None else scores_fb
            num = num if num is not None else num_fb
        except Exception:
            pass

    # Safety checks
    if boxes is None or classes is None or scores is None:
        raise RuntimeError("Unable to parse model outputs (boxes/classes/scores).")
    if num is None:
        num = min(len(scores), len(boxes), len(classes))
    else:
        num = min(num, len(scores), len(boxes), len(classes))
    return boxes, classes, scores, num

def draw_detections(frame, boxes, classes, scores, num, labels, threshold):
    h, w = frame.shape[:2]
    for i in range(num):
        score = float(scores[i])
        if score < threshold:
            continue

        # boxes are in [ymin, xmin, ymax, xmax], normalized to [0,1]
        y_min, x_min, y_max, x_max = boxes[i]
        left = int(max(0, min(w - 1, x_min * w)))
        top = int(max(0, min(h - 1, y_min * h)))
        right = int(max(0, min(w - 1, x_max * w)))
        bottom = int(max(0, min(h - 1, y_max * h)))

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label
        class_id = int(classes[i])
        label_text = get_label_text(labels, class_id)
        caption = "{} {:.2f}".format(label_text, score)

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(top - 10, text_h + 2)
        cv2.rectangle(frame, (left, y_text - text_h - 2), (left + text_w + 2, y_text + baseline - 2), color, thickness=-1)
        cv2.putText(frame, caption, (left + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    # Setup: Interpreter and labels
    if not os.path.exists(MODEL_PATH):
        print("Model file not found at:", MODEL_PATH)
        return
    labels = load_labels(LABEL_PATH)

    interpreter = make_interpreter(MODEL_PATH)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    # Expect shape [1, height, width, channels]
    in_h, in_w = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    input_index = input_details[0]['index']

    # Setup: Video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Failed to open input video:", INPUT_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print("Failed to open output video for writing:", OUTPUT_PATH)
        cap.release()
        return

    # Processing loop
    frame_count = 0
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess_frame(frame, (in_h, in_w), input_dtype)

        # Inference
        interpreter.set_tensor(input_index, inp)
        t0 = time.time()
        interpreter.invoke()
        infer_time_ms = (time.time() - t0) * 1000.0

        # Outputs
        boxes, classes, scores, num = find_output_tensors(interpreter)

        # Draw results
        draw_detections(frame, boxes, classes, scores, num, labels, CONFIDENCE_THRESHOLD)

        # Optional: overlay inference time
        info_text = "Inference: {:.1f} ms".format(infer_time_ms)
        cv2.putText(frame, info_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_count += 1

    # Cleanup
    cap.release()
    writer.release()
    total_time = time.time() - t_start
    if total_time > 0 and frame_count > 0:
        print("Processed {} frames in {:.2f}s ({:.2f} FPS).".format(frame_count, total_time, frame_count / total_time))
    print("Output saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()