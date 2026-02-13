import os
import time
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: Interpreter per guideline
from ai_edge_litert.interpreter import Interpreter

def load_labels(label_path):
    labels = []
    if os.path.isfile(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    return labels

def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_output_mappings(interpreter, output_details):
    """
    Attempts to identify SSD-like TFLite outputs:
    - boxes: [1, num, 4] ymin, xmin, ymax, xmax
    - classes: [1, num]
    - scores: [1, num]
    - num_detections: [1] or [1,1]
    Returns a dict with numpy arrays for 'boxes', 'classes', 'scores', 'num'.
    """
    outputs = [interpreter.get_tensor(od['index']) for od in output_details]
    boxes = None
    classes = None
    scores = None
    num = None

    # Identify boxes by last dimension being 4
    for out in outputs:
        arr = np.squeeze(out)
        if arr.ndim == 2 and arr.shape[-1] == 4:
            boxes = out
            break

    # Identify num detections by scalar or [1] or [1,1]
    for out in outputs:
        arr = np.squeeze(out)
        if arr.size == 1:
            try:
                num = int(np.round(float(arr)))
            except Exception:
                num = int(arr)
            break

    # Identify scores vs classes among remaining outputs
    for out in outputs:
        if boxes is not None and out is boxes:
            continue
        arr = np.squeeze(out)
        if arr.size == 1 and num is not None and int(np.round(float(arr))) == num:
            # likely num already taken
            continue
        # Scores typically in [0,1]
        if np.issubdtype(out.dtype, np.floating):
            mn = float(np.min(out))
            mx = float(np.max(out))
            if 0.0 <= mn and mx <= 1.0:
                scores = out
            else:
                # could be classes as float IDs
                if classes is None:
                    classes = out
        else:
            # integer or other => likely classes
            if classes is None:
                classes = out

    # Fallbacks: if classes or scores are still None, try remaining
    if scores is None or classes is None:
        for out in outputs:
            if out is boxes:
                continue
            if num is not None and np.squeeze(out).size == 1:
                continue
            if scores is None and np.issubdtype(out.dtype, np.floating):
                scores = out
            elif classes is None:
                classes = out

    return {
        'boxes': boxes,
        'classes': classes,
        'scores': scores,
        'num': num
    }

def draw_label_with_background(img, text, left, top, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1, color=(255,255,255), bg_color=(0,0,0)):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (left, top - text_h - baseline), (left + text_w, top + baseline), bg_color, thickness=cv2.FILLED)
    cv2.putText(img, text, (left, top - baseline), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

def color_for_class(class_id):
    # Deterministic color per class
    b = (37 * (class_id + 1)) % 255
    g = (17 * (class_id + 1)) % 255
    r = (29 * (class_id + 1)) % 255
    return int(b), int(g), int(r)

def get_label_name(class_id, labels):
    # Try both 0-based and 1-based indexing robustly
    if labels:
        if 0 <= class_id < len(labels):
            return labels[class_id]
        elif 0 <= class_id - 1 < len(labels):
            return labels[class_id - 1]
    return f'class_{class_id}'

def compute_map_proxy(class_scores_dict):
    """
    Proxy mAP calculation:
    True mAP requires ground truth annotations. Since none are provided,
    this function computes a proxy metric by:
      - averaging detection scores per class across the video
      - taking the mean over classes that had detections
    This is NOT a true mAP, but a confidence-based proxy for demonstration.
    """
    per_class_means = []
    for scores in class_scores_dict.values():
        if len(scores) > 0:
            per_class_means.append(float(np.mean(scores)))
    if not per_class_means:
        return 0.0
    return float(np.mean(per_class_means))

def main():
    # 1.2 Paths/Parameters
    model_path = 'models/ssd-mobilenet_v1/detect.tflite'
    label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
    input_path = 'data/object_detection/sheeps.mp4'
    output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
    confidence_threshold = float('0.5')

    # Ensure output directory exists
    ensure_dir_exists(output_path)

    # 1.3 Load Labels
    labels = load_labels(label_path)

    # 1.4 Load Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        raise RuntimeError("No input details available from the TFLite interpreter.")
    if not output_details:
        raise RuntimeError("No output details available from the TFLite interpreter.")

    input_shape = input_details[0]['shape']
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire input data: Read a single video file from input_path
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    # Prepare video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for output: {output_path}")

    # Stats for proxy mAP
    class_scores_dict = {}  # class_id -> list of scores
    total_frames = 0
    inference_times = []

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # End of video
            total_frames += 1

            # 2.2 Preprocess Data to match model input shape/dtype
            # Convert BGR to RGB, resize, add batch dimension
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
            input_data = np.expand_dims(resized, axis=0)

            # 2.3 Quantization Handling
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = np.uint8(input_data)

            # Phase 3: Inference
            # 3.1 Set Input Tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            # 3.2 Run Inference
            t0 = time.time()
            interpreter.invoke()
            t1 = time.time()
            inference_times.append(t1 - t0)

            # Phase 4: Output Interpretation & Handling
            # 4.1 Get Output Tensors
            out_map = get_output_mappings(interpreter, output_details)
            boxes = out_map['boxes']
            classes = out_map['classes']
            scores = out_map['scores']
            num = out_map['num']

            if boxes is None or classes is None or scores is None:
                raise RuntimeError("Failed to parse model outputs (boxes/classes/scores).")

            # Squeeze to expected shapes
            boxes = np.squeeze(boxes)  # [num, 4]
            classes = np.squeeze(classes)  # [num]
            scores = np.squeeze(scores)  # [num]
            if boxes.ndim == 1 and boxes.shape[0] == 4:
                boxes = np.expand_dims(boxes, axis=0)
            if classes.ndim == 0:
                classes = np.expand_dims(classes, axis=0)
            if scores.ndim == 0:
                scores = np.expand_dims(scores, axis=0)

            if num is None:
                # Fallback: use min shape across outputs
                det_count = min(boxes.shape[0], classes.shape[0], scores.shape[0])
            else:
                det_count = min(int(num), boxes.shape[0], classes.shape[0], scores.shape[0])

            # 4.2 Interpret Results: map class indices to labels and prepare display
            # 4.3 Post-processing: thresholding, coordinate scaling, clipping
            for i in range(det_count):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue

                box = boxes[i]  # [ymin, xmin, ymax, xmax] normalized 0..1
                ymin = max(0, min(int(round(box[0] * frame_h)), frame_h - 1))
                xmin = max(0, min(int(round(box[1] * frame_w)), frame_w - 1))
                ymax = max(0, min(int(round(box[2] * frame_h)), frame_h - 1))
                xmax = max(0, min(int(round(box[3] * frame_w)), frame_w - 1))

                # Fix potential inverted boxes
                if xmax < xmin:
                    xmin, xmax = xmax, xmin
                if ymax < ymin:
                    ymin, ymax = ymax, ymin

                class_id = int(round(float(classes[i])))
                label_name = get_label_name(class_id, labels)
                color = color_for_class(class_id)

                # Draw rectangle and label
                cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, thickness=2)
                label_text = f"{label_name} {score:.2f}"
                draw_label_with_background(frame_bgr, label_text, xmin, max(ymin, 20), color=(255,255,255), bg_color=color)

                # Update stats for proxy mAP
                if class_id not in class_scores_dict:
                    class_scores_dict[class_id] = []
                class_scores_dict[class_id].append(score)

            # Compute running proxy mAP and overlay
            map_proxy = compute_map_proxy(class_scores_dict)
            # Overlay mAP (proxy) and FPS on the frame
            avg_infer_time = np.mean(inference_times) if len(inference_times) > 0 else 0.0
            fps_text = f"FPS: {1.0/avg_infer_time:.1f}" if avg_infer_time > 0 else "FPS: N/A"
            # mAP label: clarified as proxy due to lack of ground truth
            map_text = f"mAP (proxy): {map_proxy:.3f}"
            draw_label_with_background(frame_bgr, map_text, 10, 30, font_scale=0.6, thickness=2, color=(255,255,255), bg_color=(0,0,0))
            draw_label_with_background(frame_bgr, fps_text, 10, 60, font_scale=0.6, thickness=2, color=(255,255,255), bg_color=(0,0,0))

            # 4.4 Handle Output: write annotated frame to output video
            writer.write(frame_bgr)

            # 4.5 Loop Continuation: auto-continues until video ends

    finally:
        # Phase 5: Cleanup
        cap.release()
        writer.release()

    # Final reporting
    final_map_proxy = compute_map_proxy(class_scores_dict)
    print("Processing complete.")
    print(f"Output video saved to: {output_path}")
    print(f"Frames processed: {total_frames}")
    print(f"Average inference time per frame: {np.mean(inference_times)*1000.0:.2f} ms" if inference_times else "Average inference time per frame: N/A")
    print(f"Calculated mAP (proxy, no ground truth provided): {final_map_proxy:.4f}")
    if labels:
        print("First 10 labels loaded:")
        for i, name in enumerate(labels[:10]):
            print(f"  {i}: {name}")

if __name__ == "__main__":
    """
    Application: Object Detection via a video file
    Target device: Raspberry Pi 4B
    - Reads single video file from input_path
    - Runs TFLite SSD model via ai_edge_litert Interpreter
    - Outputs annotated video with bounding boxes, labels, and overlays proxy mAP
    Note:
    True mAP requires ground-truth annotations, which are not provided.
    A confidence-based proxy mAP is computed and displayed for demonstration.
    """
    main()