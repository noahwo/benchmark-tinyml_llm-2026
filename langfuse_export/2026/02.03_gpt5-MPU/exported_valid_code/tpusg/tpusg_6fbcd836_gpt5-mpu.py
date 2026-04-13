import os
import time
import numpy as np

# Phase 1: Setup

# 1.1: Imports with fallback for Interpreter and load_delegate
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tflite_source = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        tflite_source = "tensorflow.lite"
    except Exception as e:
        raise RuntimeError("Failed to import TFLite Interpreter. Ensure tflite_runtime or TensorFlow Lite is installed.") from e

# Import cv2 only because image/video processing is explicitly required
import cv2


def load_labels(label_file_path):
    labels = []
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                clean = line.strip()
                if clean:
                    labels.append(clean)
    except Exception as e:
        print(f"WARNING: Failed to read labels from {label_file_path}. Error: {e}")
    return labels


def create_interpreter_with_edgetpu(model_file_path):
    # Try to load EdgeTPU delegate from common locations, fallback to CPU with informative messages.
    delegate_paths = [
        'libedgetpu.so.1.0',
        '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0',
    ]
    last_error = None
    for dpath in delegate_paths:
        try:
            interpreter = Interpreter(model_path=model_file_path,
                                      experimental_delegates=[load_delegate(dpath)])
            interpreter.allocate_tensors()
            print(f"INFO: Loaded model with EdgeTPU delegate '{dpath}' using {tflite_source}.")
            return interpreter, True
        except Exception as e:
            last_error = e
            continue

    print("ERROR: Failed to load EdgeTPU delegate from known paths.")
    if last_error:
        print(f"Last delegate error: {last_error}")
    print("FALLBACK: Proceeding with CPU-only TFLite interpreter. Performance will be reduced.")
    try:
        interpreter = Interpreter(model_path=model_file_path)
        interpreter.allocate_tensors()
        print(f"INFO: Loaded model without EdgeTPU delegate using {tflite_source}.")
        return interpreter, False
    except Exception as e:
        raise RuntimeError(f"Failed to load TFLite model from path '{model_file_path}'. Error: {e}") from e


def get_output_tensors(interpreter, output_details):
    outputs = []
    for od in output_details:
        outputs.append(interpreter.get_tensor(od['index']))
    return outputs


def map_outputs(outputs):
    # Attempt to infer boxes, classes, scores, and num_detections from raw output tensors.
    boxes = None
    classes = None
    scores = None
    num = None

    for out in outputs:
        arr = out
        if arr.ndim == 3 and arr.shape[-1] == 4:
            # Typically [1, num, 4]
            boxes = arr[0]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            # Could be classes or scores; determine by value range
            max_val = np.max(arr)
            min_val = np.min(arr)
            if arr.dtype == np.float32 and 0.0 <= min_val and max_val <= 1.0:
                scores = arr[0]
            else:
                # Classes often float but not constrained to [0,1]
                classes = arr[0]
        elif arr.ndim == 1 and arr.shape[0] == 1:
            num = int(arr[0])

    # Sanity checks
    if boxes is None or classes is None or scores is None or num is None:
        # Fallback attempt: handle typical ordering [boxes, classes, scores, num]
        try:
            b, c, s, n = outputs
            if b.ndim == 3 and b.shape[-1] == 4:
                boxes = b[0]
            classes = c[0] if c.ndim == 2 else c
            scores = s[0] if s.ndim == 2 else s
            num = int(n[0]) if n.ndim == 1 else int(n.squeeze())
        except Exception:
            raise RuntimeError("Unable to parse TFLite detection outputs. Unexpected output tensor shapes.")

    return boxes, classes, scores, num


def draw_bounding_box(frame, box, label_text, color=(0, 255, 0), thickness=2):
    ymin, xmin, ymax, xmax = box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
    # Draw label background
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top_left = (xmin, max(0, ymin - text_height - baseline - 4))
    bottom_right = (xmin + text_width + 6, ymin)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness=-1)
    cv2.putText(frame, label_text, (xmin + 3, ymin - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    # 1.2: Paths/Parameters
    model_path = '/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite'
    label_path = '/home/mendel/tinyml_autopilot/models/labelmap.txt'
    input_path = '/home/mendel/tinyml_autopilot/data//sheeps.mp4'
    output_path = '/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4'
    confidence_threshold = '0.5'

    # Convert confidence threshold to float
    try:
        conf_thresh = float(confidence_threshold)
    except Exception:
        conf_thresh = 0.5

    # 1.3: Load Labels
    labels = load_labels(label_path)

    # 1.4: Load Interpreter with EdgeTPU
    interpreter, using_tpu = create_interpreter_with_edgetpu(model_path)

    # 1.5: Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # [1, height, width, channels]
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    if len(input_shape) != 4 or input_shape[-1] not in (1, 3):
        raise RuntimeError(f"Unexpected model input shape: {input_shape}")

    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    input_channels = int(input_shape[3])

    # Phase 2: Input Acquisition & Preprocessing Loop

    # 2.1: Acquire Input Data (video file)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video at path: {input_path}")

    # Prepare output video writer
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps <= 0 or np.isnan(in_fps):
        in_fps = 25.0  # default fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_writer = cv2.VideoWriter(output_path, fourcc, in_fps, (frame_w, frame_h))
    if not out_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing at path: {output_path}")

    # For mAP proxy calculation across the video
    class_to_scores = {}  # class_id -> list of confidence scores
    total_frames = 0
    t_start = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            total_frames += 1

            # 2.2: Preprocess Data
            # Convert BGR to RGB (most TFLite detection models expect RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize to model input size
            resized = cv2.resize(frame_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

            # Ensure proper channel handling
            if input_channels == 1:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                resized = np.expand_dims(resized, axis=-1)

            input_data = np.expand_dims(resized, axis=0)

            # 2.3: Quantization Handling
            if floating_model:
                # Normalize to [-1, 1]
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = np.asarray(input_data, dtype=input_dtype)

            # Phase 3: Inference

            # 3.1: Set Input Tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # 3.2: Run Inference
            interpreter.invoke()

            # Phase 4: Output Interpretation & Handling Loop

            # 4.1: Get Output Tensor(s)
            raw_outputs = get_output_tensors(interpreter, output_details)

            # 4.2: Interpret Results
            boxes, classes, scores, num = map_outputs(raw_outputs)

            # 4.3: Post-processing
            # Apply confidence thresholding, coordinate scaling, and clipping
            detections_this_frame = 0
            for i in range(num):
                score = float(scores[i])
                if score < conf_thresh:
                    continue

                # Update proxy mAP accumulation
                cls_id = int(classes[i])
                if cls_id not in class_to_scores:
                    class_to_scores[cls_id] = []
                class_to_scores[cls_id].append(score)

                # Scale box coordinates to original frame size
                # boxes are typically [ymin, xmin, ymax, xmax] normalized [0,1]
                ymin = max(0, min(1.0, float(boxes[i][0])))
                xmin = max(0, min(1.0, float(boxes[i][1])))
                ymax = max(0, min(1.0, float(boxes[i][2])))
                xmax = max(0, min(1.0, float(boxes[i][3])))

                x1 = int(xmin * frame_w)
                y1 = int(ymin * frame_h)
                x2 = int(xmax * frame_w)
                y2 = int(ymax * frame_h)

                # Clip to frame boundaries
                x1 = max(0, min(frame_w - 1, x1))
                y1 = max(0, min(frame_h - 1, y1))
                x2 = max(0, min(frame_w - 1, x2))
                y2 = max(0, min(frame_h - 1, y2))

                # Prepare label text
                label_name = f"id_{cls_id}"
                if labels and 0 <= cls_id < len(labels):
                    label_name = labels[cls_id]
                label_text = f"{label_name} {score*100:.1f}%"

                # Draw rectangle and label on original BGR frame
                draw_bounding_box(frame_bgr, (y1, x1, y2, x2), label_text, color=(0, 255, 0), thickness=2)
                detections_this_frame += 1

            # Compute proxy mAP across seen classes so far:
            # mAP_proxy = mean of per-class average confidence across classes that had detections
            if class_to_scores:
                per_class_avg = [float(np.mean(sc)) for sc in class_to_scores.values() if len(sc) > 0]
                map_proxy = float(np.mean(per_class_avg)) if per_class_avg else 0.0
            else:
                map_proxy = 0.0

            # 4.4: Handle Output (write frame with overlay)
            # Overlay inference info
            runtime_tag = "EdgeTPU" if using_tpu else "CPU"
            info_text_1 = f"{runtime_tag} | Conf Thresh: {conf_thresh:.2f} | Dets: {detections_this_frame}"
            info_text_2 = f"mAP: {map_proxy:.3f}"

            cv2.putText(frame_bgr, info_text_1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 230), 2)
            cv2.putText(frame_bgr, info_text_2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 230), 2)

            out_writer.write(frame_bgr)

        # End of video loop

    finally:
        # Phase 5: Cleanup
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    fps_eff = (total_frames / elapsed) if elapsed > 0 else 0.0

    # Final mAP (proxy) calculation across entire video
    if class_to_scores:
        per_class_avg = [float(np.mean(sc)) for sc in class_to_scores.values() if len(sc) > 0]
        final_map_proxy = float(np.mean(per_class_avg)) if per_class_avg else 0.0
    else:
        final_map_proxy = 0.0

    print("========== Inference Summary ==========")
    print(f"Frames processed: {total_frames}")
    print(f"Total time (s): {elapsed:.2f}")
    print(f"Effective FPS: {fps_eff:.2f}")
    print(f"Calculated mAP (proxy): {final_map_proxy:.4f}")
    print(f"Output video saved to: {output_path}")


if __name__ == "__main__":
    main()