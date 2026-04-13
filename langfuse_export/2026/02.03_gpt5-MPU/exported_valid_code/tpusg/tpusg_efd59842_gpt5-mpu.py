import os
import sys
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports (Interpreter and delegate with fallback)
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    from tensorflow.lite import Interpreter  # type: ignore
    from tensorflow.lite.experimental import load_delegate  # type: ignore

# Import cv2 only because image/video processing is explicitly required
import cv2

# 1.2 Paths/Parameters (use provided paths/parameters exactly)
MODEL_PATH  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
OUTPUT_PATH  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
CONFIDENCE_THRESHOLD  = 0.5

# 1.3 Load Labels (Conditional)
def load_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                labels.append(line)
    return labels

# 1.4 Load Interpreter with EdgeTPU
def load_interpreter_with_edgetpu(model_path):
    last_err_msgs = []
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
        return interpreter
    except Exception as e1:
        last_err_msgs.append(f"Attempt with 'libedgetpu.so.1.0' failed: {e1}")
        try:
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
            )
            return interpreter
        except Exception as e2:
            last_err_msgs.append(f"Attempt with '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0' failed: {e2}")
            err_message = (
                "ERROR: Failed to load EdgeTPU delegate for acceleration.\n" +
                "\n".join(last_err_msgs) +
                "\nEnsure the EdgeTPU runtime is installed and the EdgeTPU device is connected.\n"
                "On Coral Dev Board, libedgetpu should be available. Refer to Coral setup docs."
            )
            print(err_message)
            sys.exit(1)

# 4.1 Get Output Tensor(s)
def get_outputs(interpreter, output_details):
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    boxes = None
    classes = None
    scores = None
    count = None

    # Typical EdgeTPU SSD order: [boxes, classes, scores, count]
    if len(outputs) >= 3:
        # Heuristic mapping by shapes/dtypes
        for i, od in enumerate(output_details):
            shp = od['shape']
            if len(shp) == 3 and shp[-1] == 4:
                boxes = outputs[i][0]
        cand = []
        for i, od in enumerate(output_details):
            shp = od['shape']
            if len(shp) == 2 and shp[0] == 1:
                cand.append((i, outputs[i][0], od))
        for i, arr, od in cand:
            if arr.dtype == np.float32:
                if scores is None:
                    scores = arr
                else:
                    maxv = float(np.nanmax(arr)) if arr.size > 0 else 0.0
                    minv = float(np.nanmin(arr)) if arr.size > 0 else 0.0
                    if minv >= 0.0 and maxv <= 1.0:
                        scores = arr
            else:
                classes = arr
        for i, od in enumerate(output_details):
            shp = od['shape']
            if len(shp) == 1 and shp[0] == 1:
                try:
                    count = int(outputs[i][0])
                except Exception:
                    pass

    if boxes is None and len(outputs) >= 1:
        try:
            boxes = outputs[0][0]
        except Exception:
            pass
    if classes is None and len(outputs) >= 2:
        try:
            classes = outputs[1][0]
        except Exception:
            pass
    if scores is None and len(outputs) >= 3:
        try:
            scores = outputs[2][0]
        except Exception:
            pass
    if count is None and len(outputs) >= 4:
        try:
            count = int(outputs[3][0])
        except Exception:
            pass

    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if classes is None:
        classes = np.zeros((boxes.shape[0],), dtype=np.float32)
    if scores is None:
        scores = np.zeros((boxes.shape[0],), dtype=np.float32)
    if count is None:
        count = min(len(scores), len(boxes))

    n = min(count, len(scores), len(classes), len(boxes))
    boxes = boxes[:n]
    classes = classes[:n]
    scores = scores[:n]
    return boxes, classes, scores, n

def iou(boxA, boxB):
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
    denom = float(boxAArea + boxBArea - interArea + 1e-6)
    return interArea / denom

def nms_per_class(detections, iou_threshold=0.5):
    by_class = {}
    for det in detections:
        by_class.setdefault(det['class_id'], []).append(det)
    kept = []
    for cls, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        selected = []
        while dets_sorted:
            current = dets_sorted.pop(0)
            selected.append(current)
            dets_sorted = [d for d in dets_sorted if iou(current['box'], d['box']) < iou_threshold]
        kept.extend(selected)
    return kept

def draw_bounding_box(frame, box, label_text, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y1, th + 2)
    cv2.rectangle(frame, (x1, y_text - th - 2), (x1 + tw + 2, y_text + baseline - 2), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def compute_map_proxy(per_class_scores):
    ap_values = []
    for cls_id, scores in per_class_scores.items():
        if len(scores) > 0:
            ap_values.append(float(np.mean(scores)))
    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))

def main():
    # 1.3 Load Labels (if provided and relevant)
    if not os.path.exists(LABEL_PATH):
        print(f"ERROR: Label file not found at: {LABEL_PATH}")
        sys.exit(1)
    labels = load_labels(LABEL_PATH)

    # 1.4 Load interpreter with EdgeTPU
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at: {MODEL_PATH}")
        sys.exit(1)
    interpreter = load_interpreter_with_edgetpu(MODEL_PATH)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)
    in_h, in_w = int(input_shape[1]), int(input_shape[2])

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input video not found at: {INPUT_PATH}")
        sys.exit(1)
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Unable to open video: {INPUT_PATH}")
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # Fallback FPS

    # Prepare writer
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        print(f"ERROR: Unable to create output video: {OUTPUT_PATH}")
        cap.release()
        sys.exit(1)

    # Aggregators for mAP proxy
    per_class_scores = {}
    total_frames = 0
    iou_threshold_nms = 0.5

    # 2.4 Loop Control: process single video file frame-by-frame
    start_time_all = time.time()
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        total_frames += 1
        frame = frame_bgr.copy()

        # 2.2 Preprocess Data
        frame_resized = cv2.resize(frame_bgr, (in_w, in_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(frame_rgb, axis=0)

        # 2.3 Quantization Handling
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = input_data.astype(input_dtype)

        # Phase 3: Inference
        # 3.1 Set Input Tensor(s)
        interpreter.set_tensor(input_index, input_data)
        # 3.2 Run Inference
        inference_start = time.time()
        interpreter.invoke()
        inference_end = time.time()

        # Phase 4: Output Interpretation & Handling Loop
        # 4.1 Get Output Tensor(s)
        boxes, classes, scores, count = get_outputs(interpreter, output_details)

        # 4.2 Interpret Results
        detections = []
        for i in range(count):
            score = float(scores[i])
            if score < CONFIDENCE_THRESHOLD:
                continue

            # Boxes are expected as [ymin, xmin, ymax, xmax] normalized to [0,1]
            y_min, x_min, y_max, x_max = boxes[i]
            x1 = int(max(0.0, min(1.0, float(x_min))) * orig_w)
            y1 = int(max(0.0, min(1.0, float(y_min))) * orig_h)
            x2 = int(max(0.0, min(1.0, float(x_max))) * orig_w)
            y2 = int(max(0.0, min(1.0, float(y_max))) * orig_h)

            # Clip and correct box order
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1 = max(0, min(x1, orig_w - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            y2 = max(0, min(y2, orig_h - 1))
            box_px = [x1, y1, x2, y2]

            cls_raw = classes[i]
            try:
                class_id = int(cls_raw)
            except Exception:
                class_id = int(float(cls_raw))

            # Map class_id to label if available
            if 0 <= class_id < len(labels):
                label_name = labels[class_id]
            else:
                label_name = f"class_{class_id}"

            detections.append({
                'class_id': class_id,
                'label': label_name,
                'score': score,
                'box': box_px
            })

        # 4.3 Post-processing: NMS per class and valid box clipping already applied
        detections_nms = nms_per_class(detections, iou_threshold=iou_threshold_nms)

        # Update proxy mAP accumulators
        for det in detections_nms:
            cid = det['class_id']
            per_class_scores.setdefault(cid, []).append(det['score'])

        # Draw detections
        for det in detections_nms:
            label_text = f"{det['label']} {det['score']:.2f}"
            draw_bounding_box(frame, det['box'], label_text, color=(0, 255, 0), thickness=2)

        # Overlay runtime info
        map_proxy_running = compute_map_proxy(per_class_scores)
        inf_ms = (inference_end - inference_start) * 1000.0
        status_text = f"mAP (proxy): {map_proxy_running:.3f} | Inference: {inf_ms:.1f} ms"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Detections: {len(detections_nms)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 240, 10), 2, cv2.LINE_AA)

        # 4.4 Handle Output: write frame to output video
        writer.write(frame)

    # Final reporting
    final_map_proxy = compute_map_proxy(per_class_scores)
    elapsed = time.time() - start_time_all
    avg_fps = (total_frames / elapsed) if elapsed > 0 else 0.0  # Precompute to avoid f-string format specifier issue
    print("INFO: Ground truth annotations not provided; reporting proxy mAP based on detection confidences.")
    print(f"Processed frames: {total_frames}")
    print(f"Total time: {elapsed:.2f} s, Avg FPS: {avg_fps:.2f}")
    print(f"Final mAP (proxy): {final_map_proxy:.4f}")
    print(f"Output video saved to: {OUTPUT_PATH}")

    # Phase 5: Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# FIX NOTE:
# The previous error "ValueError: Invalid format specifier" was caused by using an f-string with a conditional
# expression directly inside the format specifier. This script fixes it by precomputing avg_fps as a float and
# then formatting it with '{avg_fps:.2f}' in the print statement.