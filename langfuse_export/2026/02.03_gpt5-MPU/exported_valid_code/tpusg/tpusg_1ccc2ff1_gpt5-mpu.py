import os
import time
import numpy as np

# Phase 1: Setup

# 1.1 Imports with fallback (Interpreter and EdgeTPU delegate)
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    runtime_source = "tflite_runtime"
except ImportError:
    try:
        # Fallback to TensorFlow Lite (if available)
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        runtime_source = "tensorflow.lite"
    except Exception as e:
        raise SystemExit(
            "ERROR: Neither 'tflite_runtime' nor 'tensorflow.lite' could be imported. "
            "Install 'tflite-runtime' for the Google Coral Dev Board.\n"
            f"Underlying error: {e}"
        )

# 1.2 Paths/Parameters (strictly as provided)
model_path  = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path  = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path  = "/home/mendel/tinyml_autopilot/data//sheeps.mp4"
output_path  = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 1.3 Load Labels (if relevant)
def load_labels(path):
    labels = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line != "":
                    labels.append(line)
    except Exception as e:
        raise SystemExit(f"ERROR: Failed to load labels from '{path}'. Underlying error: {e}")
    return labels

labels = load_labels(label_path)

# 1.4 Load Interpreter with EdgeTPU delegate
interpreter = None
delegate_load_error_msgs = []
try:
    # First common name for EdgeTPU shared library
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
    )
except Exception as e1:
    delegate_load_error_msgs.append(f"Attempt 1 (libedgetpu.so.1.0) failed: {e1}")
    try:
        # Fallback absolute path on aarch64 platforms
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
        )
    except Exception as e2:
        delegate_load_error_msgs.append(f"Attempt 2 (/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0) failed: {e2}")
        msg = (
            "ERROR: Failed to load EdgeTPU delegate. The application is optimized for the Google Coral EdgeTPU and "
            "requires the EdgeTPU runtime. Please ensure the EdgeTPU runtime is installed and the device is connected.\n"
            f"Interpreter source: {runtime_source}\n" +
            "\n".join(delegate_load_error_msgs)
        )
        raise SystemExit(msg)

# Allocate tensors
try:
    interpreter.allocate_tensors()
except Exception as e:
    raise SystemExit(f"ERROR: Interpreter.allocate_tensors() failed. Underlying error: {e}")

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if len(input_details) < 1:
    raise SystemExit("ERROR: Model has no input tensors.")

# Input tensor properties
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Extract expected input height/width
# Typically shape is [1, height, width, 3]
if len(input_shape) != 4 or input_shape[0] != 1:
    raise SystemExit(f"ERROR: Unexpected input tensor shape: {input_shape}. Expected [1, H, W, C].")

in_height = int(input_shape[1])
in_width = int(input_shape[2])

# Determine if model is floating point
floating_model = (input_dtype == np.float32)

# Try to import cv2 only if needed (video I/O and drawing)
try:
    import cv2
except ImportError as e:
    raise SystemExit("ERROR: OpenCV (cv2) is required for video processing but is not installed. "
                     "Install OpenCV Python package.\n"
                     f"Underlying error: {e}")

# Phase 2: Input Acquisition & Preprocessing Loop

# 2.1 Acquire Input Data: open video file
video_capture = cv2.VideoCapture(input_path)
if not video_capture.isOpened():
    raise SystemExit(f"ERROR: Failed to open input video file: {input_path}")

# Retrieve input video properties
input_fps = video_capture.get(cv2.CAP_PROP_FPS)
if input_fps is None or input_fps <= 0:
    input_fps = 30.0  # fallback to a reasonable default

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_width <= 0 or frame_height <= 0:
    # Fallback by reading one frame to infer size
    ret_tmp, frame_tmp = video_capture.read()
    if not ret_tmp or frame_tmp is None:
        video_capture.release()
        raise SystemExit("ERROR: Could not read a frame to infer video size.")
    frame_height, frame_width = frame_tmp.shape[:2]
    # Reset capture to the beginning
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Create VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))
if not video_writer.isOpened():
    video_capture.release()
    raise SystemExit(f"ERROR: Failed to open output video file for writing: {output_path}")

# Prepare thresholds for proxy mAP calculation (0.5:0.95 with step 0.05)
proxy_thresholds = [0.5 + 0.05 * i for i in range(10)]

# Statistics for proxy mAP
frames_processed = 0
frames_with_proposals = 0
running_map_proxy_sum = 0.0

# Helper: Map class id to label robustly
def get_label_name(class_id, labels_list):
    try:
        # Try zero-based index first
        if 0 <= class_id < len(labels_list):
            return labels_list[class_id]
        # Try one-based index fallback
        if 1 <= class_id <= len(labels_list):
            return labels_list[class_id - 1]
    except Exception:
        pass
    return f"id_{class_id}"

# Main processing loop
try:
    while True:
        # Read a frame
        ret, frame_bgr = video_capture.read()
        if not ret or frame_bgr is None:
            break  # End of video

        frames_processed += 1
        orig_h, orig_w = frame_bgr.shape[:2]

        # 2.2 Preprocess Data:
        # Convert BGR (OpenCV) to RGB as models commonly expect RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        resized = cv2.resize(frame_rgb, (in_width, in_height))
        # Add batch dimension
        input_data = np.expand_dims(resized, axis=0)

        # 2.3 Quantization Handling
        if floating_model:
            # Normalize to [-1, 1] as per common MobileNet preprocessing
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # Ensure correct dtype for quantized model (usually uint8)
            input_data = np.asarray(input_data, dtype=input_dtype)

        # Phase 3: Inference
        # 3.1 Set input tensor
        interpreter.set_tensor(input_index, input_data)
        # 3.2 Run inference
        start_invoke = time.time()
        interpreter.invoke()
        infer_time_ms = (time.time() - start_invoke) * 1000.0

        # Phase 4: Output Interpretation & Handling

        # 4.1 Get Output Tensors
        # Typical SSD models output: boxes [1, N, 4], classes [1, N], scores [1, N], num_detections [1]
        boxes = None
        classes = None
        scores = None
        num_detections = None

        for od in output_details:
            out = interpreter.get_tensor(od['index'])
            # Remove batch dimension if present
            out_squeezed = np.squeeze(out, axis=0) if out.ndim >= 2 and out.shape[0] == 1 else out

            # Identify tensor by shape characteristics
            if out_squeezed.ndim == 2 and out_squeezed.shape[1] == 4:
                boxes = out_squeezed  # shape [N, 4]
            elif out_squeezed.ndim == 1 and out_squeezed.size > 4:
                # Heuristic: scores are usually [0,1]
                max_val = float(np.max(out_squeezed)) if out_squeezed.size > 0 else 0.0
                min_val = float(np.min(out_squeezed)) if out_squeezed.size > 0 else 0.0
                if 0.0 <= min_val and max_val <= 1.0:
                    scores = out_squeezed  # shape [N]
                else:
                    classes = out_squeezed.astype(np.int32)  # shape [N]
            elif out_squeezed.ndim == 0 or (out_squeezed.ndim == 1 and out_squeezed.size == 1):
                try:
                    num_detections = int(out_squeezed)
                except Exception:
                    pass

        # Ensure arrays are available and consistent
        if boxes is None or scores is None or classes is None:
            # Some models may return outputs in a different order; try alternative indexing
            # Attempt to infer from first four output tensors if shapes match
            if len(output_details) >= 3:
                outs = [np.squeeze(interpreter.get_tensor(od['index'])) for od in output_details[:4]]
                # Attempt to set by identifying 2D (N,4)
                for arr in outs:
                    if arr.ndim == 2 and arr.shape[-1] == 4:
                        boxes = arr
                # Identify scores/classes
                one_dim = [arr for arr in outs if arr.ndim == 1 and arr.size > 1]
                for arr in one_dim:
                    max_val = float(np.max(arr))
                    min_val = float(np.min(arr))
                    if 0.0 <= min_val <= 1.0 and max_val <= 1.0:
                        scores = arr
                    else:
                        classes = arr.astype(np.int32)
            # Final check
            if boxes is None or scores is None or classes is None:
                # If outputs cannot be parsed, skip drawing for this frame but write original
                overlay_text = "Output parsing error"
                cv2.putText(frame_bgr, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                video_writer.write(frame_bgr)
                continue

        if num_detections is None:
            num_detections = min(len(scores), len(classes), boxes.shape[0])

        # 4.2 Interpret Results: apply thresholding and map labels
        # 4.3 Post-processing: scale and clip boxes
        detections_above_thresh = 0
        total_proposals = int(num_detections)

        # Prepare for proxy mAP computation
        conf_list = []
        for i in range(total_proposals):
            score = float(scores[i])
            conf_list.append(score)
        # Compute proxy AP for this frame across thresholds (0.5:0.95 step 0.05)
        if total_proposals > 0:
            per_threshold_precisions = []
            for t in proxy_thresholds:
                cnt = sum(1 for c in conf_list if c >= t)
                per_threshold_precisions.append(cnt / float(total_proposals))
            frame_map_proxy = float(np.mean(per_threshold_precisions)) if len(per_threshold_precisions) > 0 else 0.0
            running_map_proxy_sum += frame_map_proxy
            frames_with_proposals += 1
        else:
            frame_map_proxy = 0.0  # no proposals

        # Draw detections
        for i in range(total_proposals):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            detections_above_thresh += 1

            # Extract box and handle both normalized and absolute coords
            y_min, x_min, y_max, x_max = boxes[i].tolist()

            # Determine if normalized
            max_coord_val = max(abs(y_min), abs(x_min), abs(y_max), abs(x_max))
            # Assume normalized if all in [0, 1.5]
            if max_coord_val <= 1.5:
                # Clip to [0,1]
                y_min = max(0.0, min(1.0, y_min))
                x_min = max(0.0, min(1.0, x_min))
                y_max = max(0.0, min(1.0, y_max))
                x_max = max(0.0, min(1.0, x_max))
                # Scale to original frame size
                x1 = int(x_min * orig_w)
                y1 = int(y_min * orig_h)
                x2 = int(x_max * orig_w)
                y2 = int(y_max * orig_h)
            else:
                # Treat as absolute relative to model input size
                # Clip to model input bounds then scale to original frame
                y_min = max(0.0, min(float(in_height), y_min))
                x_min = max(0.0, min(float(in_width), x_min))
                y_max = max(0.0, min(float(in_height), y_max))
                x_max = max(0.0, min(float(in_width), x_max))
                x1 = int((x_min / float(in_width)) * orig_w)
                y1 = int((y_min / float(in_height)) * orig_h)
                x2 = int((x_max / float(in_width)) * orig_w)
                y2 = int((y_max / float(in_height)) * orig_h)

            # Clip to frame bounds
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w - 1, x2))
            y2 = max(0, min(orig_h - 1, y2))

            class_id = int(classes[i])
            label_name = get_label_name(class_id, labels)
            box_color = (0, 255, 0)  # Green for boxes
            text_color = (0, 0, 0)   # Black text
            text_bg_color = (0, 255, 255)  # Yellow background for text

            # Draw rectangle
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, 2)

            # Prepare label text
            caption = f"{label_name}: {score:.2f}"

            # Draw filled rectangle for text background
            (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10 + text_h
            cv2.rectangle(frame_bgr, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y + 2), text_bg_color, -1)
            cv2.putText(frame_bgr, caption, (text_x + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Overlay runtime and mAP proxy
        status_color = (255, 255, 255)
        info_y = 20
        cv2.putText(frame_bgr, f"Inference: {infer_time_ms:.1f} ms", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        info_y += 25
        cv2.putText(frame_bgr, f"Detections >= {confidence_threshold:.2f}: {detections_above_thresh}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        info_y += 25
        if frames_with_proposals > 0:
            running_map_proxy = running_map_proxy_sum / float(frames_with_proposals)
            cv2.putText(frame_bgr, f"mAP (proxy, no GT): {running_map_proxy:.3f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        else:
            cv2.putText(frame_bgr, "mAP (proxy, no GT): N/A", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # 4.4 Handle Output: write the annotated frame
        video_writer.write(frame_bgr)

        # 4.5 Loop Continuation: continues until video ends

finally:
    # Phase 5: Cleanup
    try:
        video_capture.release()
    except Exception:
        pass
    try:
        video_writer.release()
    except Exception:
        pass

# Final summary print
if frames_with_proposals > 0:
    final_map_proxy = running_map_proxy_sum / float(frames_with_proposals)
    print(f"Processing completed. Frames processed: {frames_processed}. Proxy mAP (no GT): {final_map_proxy:.4f}")
else:
    print(f"Processing completed. Frames processed: {frames_processed}. Proxy mAP (no GT): N/A")

print(f"Output saved to: {output_path}")