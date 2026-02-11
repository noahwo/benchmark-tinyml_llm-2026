import os
import sys
import time
import numpy as np

# Phase 1: Setup
# 1.1 Imports
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    RUNTIME_SOURCE = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        from tensorflow.lite.experimental import load_delegate  # type: ignore
        RUNTIME_SOURCE = "tensorflow.lite"
    except Exception as e:
        print("ERROR: Failed to import TFLite Interpreter from both tflite_runtime and tensorflow.lite.")
        print("Details:", str(e))
        sys.exit(1)

# OpenCV is explicitly needed for video I/O and drawing
import cv2

def load_labels(label_path):
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support plain label per line
                if line.isdigit():
                    # If labelmap is of format "<id> <label>" try to parse
                    continue
                labels.append(line)
    except Exception as e:
        print(f"WARNING: Failed to load labels from {label_path}. Details: {e}")
    return labels

def resolve_output_indices(output_details):
    """
    Attempt to resolve indices for boxes, classes, scores, and num_detections
    using names and shapes heuristics.
    """
    idx_map = {"boxes": None, "classes": None, "scores": None, "num": None}
    # First try by name
    for i, od in enumerate(output_details):
        name = od.get('name', '').lower()
        shape = od.get('shape', [])
        if 'box' in name:
            idx_map["boxes"] = i
        elif 'class' in name:
            idx_map["classes"] = i
        elif 'score' in name:
            idx_map["scores"] = i
        elif 'num' in name:
            idx_map["num"] = i

    # Fallback by shape if any are missing
    if idx_map["boxes"] is None or idx_map["classes"] is None or idx_map["scores"] is None or idx_map["num"] is None:
        # Collect candidates
        for i, od in enumerate(output_details):
            shape = od.get('shape', [])
            # boxes: [1, N, 4]
            if idx_map["boxes"] is None and len(shape) == 3 and shape[-1] == 4:
                idx_map["boxes"] = i
            # classes: [1, N]
            elif idx_map["classes"] is None and len(shape) == 2 and shape[0] == 1 and shape[1] > 1:
                # Tentatively assign; might get overridden by scores
                idx_map["classes"] = i
            # scores: [1, N]
            elif idx_map["scores"] is None and len(shape) == 2 and shape[0] == 1 and shape[1] > 1:
                # If classes already set, assign the other one to scores later
                pass
            # num: [1]
            elif idx_map["num"] is None and len(shape) == 1 and shape[0] == 1:
                idx_map["num"] = i

        # If classes and scores unresolved or ambiguous, decide by dtype ranges if possible
        unresolved_two = [i for i, od in enumerate(output_details) if len(od.get('shape', [])) == 2 and od['shape'][0] == 1 and od['shape'][1] > 1]
        if (idx_map["classes"] is None or idx_map["scores"] is None) and len(unresolved_two) >= 2:
            # Heuristic: classes are integer-like floats (whole numbers), scores are floats in [0,1]
            # We can't read tensors before invoke, so we'll just assign the first to scores and second to classes if missing
            if idx_map["scores"] is None:
                idx_map["scores"] = unresolved_two[0]
            if idx_map["classes"] is None:
                idx_map["classes"] = unresolved_two[1]

    return idx_map

def clip_bbox(xmin, ymin, xmax, ymax, width, height):
    xmin = max(0, min(int(round(xmin)), width - 1))
    ymin = max(0, min(int(round(ymin)), height - 1))
    xmax = max(0, min(int(round(xmax)), width - 1))
    ymax = max(0, min(int(round(ymax)), height - 1))
    return xmin, ymin, xmax, ymax

def overlay_text(img, text, org=(10, 25), color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

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

    # 1.3 Load Labels (Conditional)
    labels = load_labels(label_path)

    # 1.4 Load Interpreter with EdgeTPU
    interpreter = None
    delegate_load_errors = []
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
    except Exception as e1:
        delegate_load_errors.append(str(e1))
        try:
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
            )
        except Exception as e2:
            delegate_load_errors.append(str(e2))
            print("ERROR: Failed to load EdgeTPU delegate. Ensure the EdgeTPU runtime is installed and the correct shared library is available.")
            print("Tried delegates: 'libedgetpu.so.1.0' and '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0'")
            print("Details:")
            for idx, err in enumerate(delegate_load_errors, 1):
                print(f"  Attempt {idx}: {err}")
            sys.exit(1)

    try:
        interpreter.allocate_tensors()
    except Exception as e:
        print("ERROR: Failed to allocate tensors for the TFLite interpreter:", str(e))
        sys.exit(1)

    # 1.5 Get Model Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details:
        print("ERROR: No input details found in the model.")
        sys.exit(1)
    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']  # Expected [1, height, width, channels]
    input_dtype = input_details[0]['dtype']
    floating_model = (input_dtype == np.float32)

    # Resolve output indices
    out_idx = resolve_output_indices(output_details)
    if None in out_idx.values():
        print("ERROR: Unable to resolve output tensor indices for boxes/classes/scores/num_detections.")
        sys.exit(1)

    # Phase 2: Input Acquisition & Preprocessing Loop
    # 2.1 Acquire Input Data
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open input video: {input_path}")
        sys.exit(1)

    # Prepare VideoWriter with input video properties
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback if FPS not available
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (input_width, input_height))
    if not writer.isOpened():
        print(f"ERROR: Failed to open output video for writing: {output_path}")
        cap.release()
        sys.exit(1)

    # Input tensor expected size
    if len(input_shape) != 4 or input_shape[0] != 1:
        print(f"ERROR: Unexpected input tensor shape: {input_shape}. Expected [1, height, width, channels].")
        cap.release()
        writer.release()
        sys.exit(1)

    model_in_height = int(input_shape[1])
    model_in_width = int(input_shape[2])

    # Variables for proxy mAP computation: collect confidences per class
    per_class_confidences = {}  # class_id -> list of confidences
    total_frames = 0
    total_detections = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            total_frames += 1
            frame_h, frame_w = frame_bgr.shape[:2]

            # 2.2 Preprocess Data
            # Resize and convert BGR->RGB
            resized = cv2.resize(frame_bgr, (model_in_width, model_in_height), interpolation=cv2.INTER_LINEAR)
            input_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_rgb, axis=0)

            # 2.3 Quantization Handling
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                if input_data.dtype != input_dtype:
                    input_data = input_data.astype(input_dtype)

            # Phase 3: Inference
            # 3.1 Set Input Tensor(s)
            interpreter.set_tensor(input_index, input_data)
            # 3.2 Run Inference
            interpreter.invoke()

            # Phase 4: Output Interpretation & Handling Loop
            # 4.1 Get Output Tensor(s)
            boxes = interpreter.get_tensor(output_details[out_idx["boxes"]]['index'])
            classes = interpreter.get_tensor(output_details[out_idx["classes"]]['index'])
            scores = interpreter.get_tensor(output_details[out_idx["scores"]]['index'])
            num = interpreter.get_tensor(output_details[out_idx["num"]]['index'])

            # Ensure expected shapes
            if boxes.ndim == 3:
                boxes = boxes[0]
            if classes.ndim == 2:
                classes = classes[0]
            if scores.ndim == 2:
                scores = scores[0]
            if num.ndim >= 1:
                num_detections = int(np.round(num.flatten()[0]))
            else:
                num_detections = len(scores)

            # 4.2 Interpret Results
            # Build overlay with bounding boxes and labels
            for i in range(num_detections):
                score = float(scores[i])
                if score < confidence_threshold:
                    continue
                total_detections += 1
                class_id = int(classes[i])
                label_text = str(class_id)
                if labels and 0 <= class_id < len(labels):
                    label_text = labels[class_id]

                # 4.3 Post-processing: thresholding, scaling, clipping
                # TFLite detection boxes are [ymin, xmin, ymax, xmax] normalized 0..1
                y_min, x_min, y_max, x_max = boxes[i]
                x_min *= frame_w
                x_max *= frame_w
                y_min *= frame_h
                y_max *= frame_h
                xmin_i, ymin_i, xmax_i, ymax_i = clip_bbox(x_min, y_min, x_max, y_max, frame_w, frame_h)

                # Draw rectangle and label
                cv2.rectangle(frame_bgr, (xmin_i, ymin_i), (xmax_i, ymax_i), (0, 255, 0), 2)
                label_draw = f"{label_text}: {score:.2f}"
                overlay_text(frame_bgr, label_draw, org=(xmin_i, max(0, ymin_i - 10)), color=(255, 255, 255), scale=0.5, thickness=2)

                # Accumulate confidences for proxy mAP computation
                if class_id not in per_class_confidences:
                    per_class_confidences[class_id] = []
                per_class_confidences[class_id].append(score)

            # Compute running proxy mAP (mean of mean confidences per class)
            if per_class_confidences:
                ap_values = [float(np.mean(confs)) for confs in per_class_confidences.values() if len(confs) > 0]
                map_proxy = float(np.mean(ap_values)) if ap_values else 0.0
            else:
                map_proxy = 0.0

            # 4.4 Handle Output: overlay and write frame
            overlay_text(frame_bgr, f"mAP: {map_proxy:.3f}", org=(10, 25), color=(0, 255, 255), scale=0.7, thickness=2)
            overlay_text(frame_bgr, f"Detections: {total_detections}", org=(10, 50), color=(0, 255, 255), scale=0.6, thickness=2)
            writer.write(frame_bgr)

        # End of loop
    finally:
        # Phase 5: Cleanup
        cap.release()
        writer.release()

    # Final proxy mAP computation and reporting
    if per_class_confidences:
        ap_values = [float(np.mean(confs)) for confs in per_class_confidences.values() if len(confs) > 0]
        final_map_proxy = float(np.mean(ap_values)) if ap_values else 0.0
    else:
        final_map_proxy = 0.0

    print("Processing completed.")
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections (>= {confidence_threshold:.2f}): {total_detections}")
    print(f"Calculated mAP: {final_map_proxy:.4f}")

if __name__ == "__main__":
    main()