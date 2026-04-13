import os
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports: TFLite Interpreter and EdgeTPU delegate with fallback
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    TFLITE_SOURCE = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        TFLITE_SOURCE = "tensorflow.lite"
    except Exception as e:
        print("ERROR: Unable to import TFLite Interpreter. Ensure tflite_runtime or tensorflow is installed.")
        raise e

# Import cv2 only because image/video processing is explicitly required
import cv2

def load_labels_file(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    labels.append(name)
    except Exception as e:
        print(f"ERROR: Failed to load label file: {label_file_path}. {e}")
        labels = []
    return labels

def create_interpreter_with_edgetpu(model_file_path):
    # Attempt to load EdgeTPU delegate from default soname, then absolute path
    delegate = None
    delegate_info = ""
    load_errors = []
    try:
        delegate = load_delegate('libedgetpu.so.1.0')
        delegate_info = 'libedgetpu.so.1.0'
    except Exception as e1:
        load_errors.append(f"Attempt 1 (libedgetpu.so.1.0): {e1}")
        try:
            delegate = load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')
            delegate_info = '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'
        except Exception as e2:
            load_errors.append(f"Attempt 2 (/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0): {e2}")
            print("ERROR: Failed to load the EdgeTPU delegate. Details:")
            for err in load_errors:
                print(f" - {err}")
            print("Please ensure the EdgeTPU runtime is installed and the device is connected.")
            return None, None

    try:
        interpreter = Interpreter(
            model_path=model_file_path,
            experimental_delegates=[delegate]
        )
        interpreter.allocate_tensors()
        return interpreter, delegate_info
    except Exception as e:
        print(f"ERROR: Failed to create or allocate TFLite Interpreter with EdgeTPU delegate ({delegate_info}). {e}")
        return None, None

def get_model_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine input specs
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    if len(input_shape) != 4:
        raise ValueError(f"Unexpected input tensor shape: {input_shape}. Expect [1, height, width, channels].")
    _, in_h, in_w, in_c = input_shape

    # Map output indices robustly by name; fallback to typical SSD order if names not informative
    boxes_idx = classes_idx = scores_idx = num_idx = None
    for i, od in enumerate(output_details):
        name = od.get('name', '').lower()
        shape = od.get('shape', [])
        if 'box' in name or (len(shape) == 3 and shape[-1] == 4):
            boxes_idx = i
        elif 'score' in name:
            scores_idx = i
        elif 'class' in name:
            classes_idx = i
        elif 'num' in name and len(shape) == 1:
            num_idx = i

    # Fallback to standard order if necessary
    if boxes_idx is None or classes_idx is None or scores_idx is None or num_idx is None:
        # Typical order for TFLite SSD models: boxes, classes, scores, num_detections
        boxes_idx = 0 if boxes_idx is None else boxes_idx
        classes_idx = 1 if classes_idx is None else classes_idx
        scores_idx = 2 if scores_idx is None else scores_idx
        num_idx = 3 if num_idx is None else num_idx

    return {
        'input_index': input_index,
        'input_height': in_h,
        'input_width': in_w,
        'input_channels': in_c,
        'input_dtype': input_dtype,
        'output_indices': {
            'boxes': boxes_idx,
            'classes': classes_idx,
            'scores': scores_idx,
            'num': num_idx
        },
        'input_details': input_details,
        'output_details': output_details
    }

def compute_map_proxy(class_confidences_dict):
    # Proxy mAP: mean of mean confidences per class that had detections
    ap_values = []
    for cls_id, confs in class_confidences_dict.items():
        if len(confs) > 0:
            ap_values.append(float(np.mean(confs)))
    if len(ap_values) == 0:
        return 0.0
    return float(np.mean(ap_values))

def main():
    # 1.2 Paths/Parameters
    model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
    label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
    input_path = '/home/mendel/tinyml_autopilot/data//sheeps.mp4'
    output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
    confidence_threshold_str = '0.5'
    try:
        confidence_threshold = float(confidence_threshold_str)
    except Exception:
        confidence_threshold = 0.5

    # 1.3 Load Labels
    labels = load_labels_file(label_path)
    if len(labels) == 0:
        print("WARNING: No labels loaded. Class IDs will be used instead of names.")

    # 1.4 Load Interpreter with EdgeTPU
    interpreter, delegate_info = create_interpreter_with_edgetpu(model_path)
    if interpreter is None:
        # Detailed errors already printed inside create_interpreter_with_edgetpu
        return
    print(f"INFO: TFLite Interpreter loaded using {TFLITE_SOURCE} with EdgeTPU delegate ({delegate_info}).")

    # 1.5 Get Model Details
    try:
        io_info = get_model_io_details(interpreter)
    except Exception as e:
        print(f"ERROR: Failed to parse model I/O details. {e}")
        return

    input_index = io_info['input_index']
    in_h = io_info['input_height']
    in_w = io_info['input_width']
    in_c = io_info['input_channels']
    input_dtype = io_info['input_dtype']
    output_indices = io_info['output_indices']
    output_details = io_info['output_details']

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data - open video file
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        print(f"ERROR: Unable to open input video file: {input_path}")
        return

    # Prepare Video Writer for output
    input_fps = video_capture.get(cv2.CAP_PROP_FPS)
    if input_fps is None or input_fps <= 0:
        input_fps = 30.0  # Default to 30 FPS if metadata missing
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        # Try to read first frame to infer size if metadata missing
        ret_probe, frame_probe = video_capture.read()
        if not ret_probe or frame_probe is None:
            print("ERROR: Failed to read from input video to determine frame size.")
            video_capture.release()
            return
        frame_height, frame_width = frame_probe.shape[:2]
        # Rewind capture to start
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
    if not video_writer.isOpened():
        print(f"ERROR: Unable to open output video file for writing: {output_path}")
        video_capture.release()
        return

    # Quantization handling setup
    floating_model = (input_dtype == np.float32)

    # Stats for mAP proxy
    class_confidences = {}  # class_id -> list of confidence scores
    total_frames = 0
    start_time = time.time()

    # Processing loop
    while True:
        # 2.1 Read next frame
        ret, frame_bgr = video_capture.read()
        if not ret or frame_bgr is None:
            break  # End of video

        total_frames += 1
        original_h, original_w = frame_bgr.shape[:2]

        # 2.2 Preprocess: resize to model input size and convert color
        resized_bgr = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        input_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        # Prepare input tensor data
        if floating_model:
            input_data = np.expand_dims(input_rgb, axis=0).astype(np.float32)
        else:
            input_data = np.expand_dims(input_rgb, axis=0).astype(np.uint8)

        # 2.3 Quantization Handling
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # Phase 3: Inference
        # 3.1 Set Input Tensor
        interpreter.set_tensor(input_index, input_data)
        # 3.2 Run Inference
        interpreter.invoke()

        # Phase 4: Output Interpretation & Handling
        # 4.1 Get Output Tensors
        boxes = interpreter.get_tensor(output_details[output_indices['boxes']]['index'])
        classes = interpreter.get_tensor(output_details[output_indices['classes']]['index'])
        scores = interpreter.get_tensor(output_details[output_indices['scores']]['index'])
        num = interpreter.get_tensor(output_details[output_indices['num']]['index'])

        # Squeeze outputs
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        # num_detections may be float or int; ensure int
        try:
            num_detections = int(np.squeeze(num).astype(np.int32))
        except Exception:
            num_detections = len(scores)

        # 4.2 Interpret Results
        # Build list of valid detections exceeding confidence threshold
        detections = []
        present_classes_in_frame = set()
        for i in range(num_detections):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            class_id = int(classes[i])
            # Map class ID to name if available
            if 0 <= class_id < len(labels):
                class_name = labels[class_id]
            else:
                class_name = f"id_{class_id}"

            # Raw normalized box: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[i].tolist()

            # 4.3 Post-processing: clip and scale to original frame size
            xmin = max(0.0, min(1.0, float(xmin)))
            ymin = max(0.0, min(1.0, float(ymin)))
            xmax = max(0.0, min(1.0, float(xmax)))
            ymax = max(0.0, min(1.0, float(ymax)))

            x1 = int(xmin * original_w)
            y1 = int(ymin * original_h)
            x2 = int(xmax * original_w)
            y2 = int(ymax * original_h)

            # Ensure valid box coordinates
            x1 = max(0, min(original_w - 1, x1))
            y1 = max(0, min(original_h - 1, y1))
            x2 = max(0, min(original_w - 1, x2))
            y2 = max(0, min(original_h - 1, y2))

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'score': score,
                'class_id': class_id,
                'class_name': class_name
            })
            present_classes_in_frame.add(class_id)

            # Update mAP proxy stats
            if class_id not in class_confidences:
                class_confidences[class_id] = []
            class_confidences[class_id].append(score)

        # 4.4 Handle Output: Draw boxes and labels, overlay mAP proxy, and write frame to output video
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            score = det['score']
            # Color based on class id
            color = (int((37 * (class_id + 1)) % 255), int((17 * (class_id + 1)) % 255), int((29 * (class_id + 1)) % 255))
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_name}: {score:.2f}"
            # Put a filled rectangle behind text for readability
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - text_h - baseline - 4)), (x1 + text_w + 4, y1), color, thickness=-1)
            cv2.putText(frame_bgr, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Compute and overlay running mAP proxy
        map_proxy_value = compute_map_proxy(class_confidences)
        map_text = f"mAP (confidence proxy): {map_proxy_value:.3f}"
        cv2.putText(frame_bgr, map_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        # Write frame to output video
        video_writer.write(frame_bgr)

    elapsed = time.time() - start_time
    final_map_proxy = compute_map_proxy(class_confidences)
    print(f"INFO: Processed {total_frames} frames in {elapsed:.2f}s ({(total_frames / max(elapsed, 1e-6)):.2f} FPS).")
    print(f"INFO: Final mAP (confidence proxy): {final_map_proxy:.4f}")
    print(f"INFO: Output saved to: {output_path}")

    # Phase 5: Cleanup
    video_capture.release()
    video_writer.release()
    # No GUI windows were opened; no need to call cv2.destroyAllWindows()

if __name__ == "__main__":
    main()