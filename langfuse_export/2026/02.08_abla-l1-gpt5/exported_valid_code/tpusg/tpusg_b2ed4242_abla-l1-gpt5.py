import os
import time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Configuration Parameters
# =========================
MODEL_PATH = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
LABEL_PATH = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
INPUT_PATH = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
INPUT_DESCRIPTION = "Read a single video file from the given input_path"
OUTPUT_PATH = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
OUTPUT_DESCRIPTION = "Output the video file with rectangles drew on the detected objects, along with texts of labels and calculated mAP(mean average precision)"
CONFIDENCE_THRESHOLD = 0.5

EDGETPU_DELEGATE_PATH = "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0"


def load_labels(path):
    """
    Loads label map file. Supports:
    - Plain list (each line a label)
    - 'id label' per line
    Returns dict: {int_id: label_str}
    """
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try "id label" format
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                # Fallback: one label per line, use line index as id
                labels[idx] = line
    return labels


def build_interpreter(model_path, delegate_path):
    """
    Create TFLite Interpreter with EdgeTPU delegate and allocate tensors.
    """
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate(delegate_path)]
    )
    interpreter.allocate_tensors()
    return interpreter


def get_input_tensor_details(interpreter):
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    input_shape = input_details['shape']  # [1, h, w, c]
    height, width = int(input_shape[1]), int(input_shape[2])
    dtype = input_details['dtype']
    quant_params = input_details.get('quantization', (0.0, 0))
    return input_index, height, width, dtype, quant_params


def get_output_tensors(interpreter):
    """
    Retrieve common detection model outputs:
    - boxes: [N,4] normalized (ymin, xmin, ymax, xmax)
    - classes: [N]
    - scores: [N]
    - num: int
    Handles variations in output order.
    """
    output_details = interpreter.get_output_details()
    boxes = classes = scores = num = None

    for od in output_details:
        tensor = interpreter.get_tensor(od['index'])
        if tensor.ndim == 3 and tensor.shape[-1] == 4:
            boxes = tensor[0]
        elif tensor.ndim == 2 and tensor.shape[-1] == 4:
            boxes = tensor[0]
        elif tensor.ndim == 2 and tensor.shape[-1] != 4:
            # Could be classes or scores
            # Heuristic: if dtype is float and values in [0,1], it's scores
            if tensor.dtype.kind == 'f':
                scores = tensor[0]
            else:
                classes = tensor[0].astype(np.int32)
        elif tensor.ndim == 1:
            # num detections or scores/classes flattened
            if tensor.size == 1:
                num = int(np.squeeze(tensor).astype(np.int32))
            else:
                # Fallback if some models return flat arrays
                if tensor.dtype.kind == 'f' and np.all((tensor >= 0) & (tensor <= 1)):
                    scores = tensor
                else:
                    classes = tensor.astype(np.int32)

    # Some models also provide quantization for outputs; dequantize if needed
    def dequantize_if_needed(detail, arr):
        if arr is None:
            return None
        q = detail.get('quantization', (0.0, 0))
        if q and isinstance(q, tuple) and q[0] not in (0.0, 1.0) and arr.dtype != np.float32:
            scale, zero_point = q
            return (arr.astype(np.float32) - zero_point) * scale
        return arr

    # Attempt to dequantize scores/boxes if needed using their respective details
    # (If float already, this is a no-op)
    # Map outputs by type for dequantization
    by_shape = {tuple(interpreter.get_tensor(od['index']).shape): od for od in output_details}
    if scores is not None:
        # find detail with matching shape to scores
        shape_key = tuple([1] + list(scores.shape)) if scores.ndim == 1 else tuple(scores.shape)
        scores_detail = by_shape.get(shape_key, output_details[0])
        scores = dequantize_if_needed(scores_detail, scores)
    if boxes is not None:
        shape_key = tuple([1] + list(boxes.shape)) if boxes.ndim == 2 else tuple(boxes.shape)
        boxes_detail = by_shape.get(shape_key, output_details[0])
        boxes = dequantize_if_needed(boxes_detail, boxes)

    # num may be provided; if not, infer from scores or boxes
    if num is None:
        if scores is not None:
            num = int(scores.shape[0])
        elif boxes is not None:
            num = int(boxes.shape[0])
        else:
            num = 0

    # Ensure classes present
    if classes is None and scores is not None:
        classes = np.zeros_like(scores, dtype=np.int32)
    if classes is None:
        classes = np.array([], dtype=np.int32)

    return boxes, classes, scores, num


def preprocess_frame_bgr_to_input(frame_bgr, input_h, input_w, input_dtype, input_quant):
    """
    Resize and convert BGR frame to model input tensor.
    Handles quantized uint8 and float32 inputs.
    """
    resized = cv2.resize(frame_bgr, (input_w, input_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if input_dtype == np.uint8:
        input_data = rgb.astype(np.uint8)
        # For quantized inputs, assume zero_point/scale handled by model; most edgetpu detect models expect uint8 [0..255]
    else:
        # float32 input assumed; normalize to [0,1]
        input_data = rgb.astype(np.float32) / 255.0

    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def put_label_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """
    Draw a filled rectangle as background for text to improve readability.
    """
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # Ensure text background box is within image bounds
    x1, y1 = max(0, x), max(0, y - th - baseline)
    x2, y2 = min(img.shape[1] - 1, x + tw), min(img.shape[0] - 1, y + baseline)
    cv2.rectangle(img, (x1, y1), (x2, y2 + th), bg_color, cv2.FILLED)
    cv2.putText(img, text, (x1, y2), font, font_scale, text_color, thickness, cv2.LINE_AA)


def get_label_name(labels_map, class_id):
    """
    Resolve label name, trying both class_id and class_id+1 (common label maps are 1-based).
    """
    if class_id in labels_map:
        return labels_map[class_id]
    elif (class_id + 1) in labels_map:
        return labels_map[class_id + 1]
    else:
        return f"id_{class_id}"


def compute_map_proxy(scores_by_class):
    """
    Compute a proxy for mAP without ground truth:
    - For each class present, AP_proxy = mean(scores)  in [0,1]
    - mAP_proxy = mean of AP_proxy across classes with detections
    Returns (mAP_proxy_float, ap_by_class_dict)
    """
    ap_by_class = {}
    for cid, scores in scores_by_class.items():
        if len(scores) == 0:
            continue
        ap_by_class[cid] = float(np.mean(scores))
    if len(ap_by_class) == 0:
        return 0.0, ap_by_class
    mAP = float(np.mean(list(ap_by_class.values())))
    return mAP, ap_by_class


def main():
    print("Application: TFLite object detection with TPU")
    print(f"Input: {INPUT_PATH} ({INPUT_DESCRIPTION})")
    print(f"Output: {OUTPUT_PATH} ({OUTPUT_DESCRIPTION})")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # Load labels
    labels = load_labels(LABEL_PATH)

    # Initialize interpreter with EdgeTPU delegate
    interpreter = build_interpreter(MODEL_PATH, EDGETPU_DELEGATE_PATH)
    input_index, in_h, in_w, in_dtype, in_quant = get_input_tensor_details(interpreter)

    # Setup video IO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open input video: {INPUT_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
    if not out.isOpened():
        print(f"ERROR: Cannot open output video for writing: {OUTPUT_PATH}")
        cap.release()
        return

    # For mAP proxy computation across frames
    scores_by_class = {}  # class_id -> list of scores in [0,1]
    total_frames = 0
    avg_infer_ms = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            total_frames += 1

            # Preprocess
            input_tensor = preprocess_frame_bgr_to_input(frame_bgr, in_h, in_w, in_dtype, in_quant)
            interpreter.set_tensor(input_index, input_tensor)

            # Inference
            t0 = time.time()
            interpreter.invoke()
            infer_ms = (time.time() - t0) * 1000.0
            # Running average of inference time
            avg_infer_ms = ((avg_infer_ms * (total_frames - 1)) + infer_ms) / total_frames

            # Postprocess outputs
            boxes, classes, scores, num = get_output_tensors(interpreter)

            # Draw detections
            drawn = frame_bgr.copy()
            det_count = 0
            if scores is not None and boxes is not None and classes is not None:
                N = int(num) if num is not None else len(scores)
                N = min(N, len(scores), len(boxes), len(classes))
                for i in range(N):
                    score = float(scores[i])
                    if score < CONFIDENCE_THRESHOLD:
                        continue
                    det_count += 1
                    cls_id = int(classes[i])
                    label_name = get_label_name(labels, cls_id)

                    # boxes are [ymin, xmin, ymax, xmax] normalized [0,1]
                    y_min, x_min, y_max, x_max = boxes[i]
                    x1 = max(0, min(frame_w - 1, int(x_min * frame_w)))
                    y1 = max(0, min(frame_h - 1, int(y_min * frame_h)))
                    x2 = max(0, min(frame_w - 1, int(x_max * frame_w)))
                    y2 = max(0, min(frame_h - 1, int(y_max * frame_h)))

                    # Draw rectangle
                    color = (0, 255, 0)
                    cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)

                    # Label text
                    text = f"{label_name}: {int(score * 100)}%"
                    put_label_with_bg(drawn, text, (x1, max(0, y1 - 5)), font_scale=0.6, bg_color=(0, 0, 0))

                    # Accumulate scores for mAP proxy
                    if cls_id not in scores_by_class:
                        scores_by_class[cls_id] = []
                    scores_by_class[cls_id].append(score)

            # Running mAP proxy
            map_proxy, _ = compute_map_proxy(scores_by_class)

            # Overlay summary
            summary_lines = [
                f"Detections: {det_count}",
                f"mAP: {map_proxy * 100:.1f}%",
                f"Infer: {infer_ms:.1f} ms (avg {avg_infer_ms:.1f} ms)"
            ]
            y0 = 20
            for line in summary_lines:
                put_label_with_bg(drawn, line, (10, y0), font_scale=0.6, bg_color=(32, 32, 32))
                y0 += 22

            # Write frame
            out.write(drawn)

    finally:
        cap.release()
        out.release()

    # Final report
    final_map, ap_by_class = compute_map_proxy(scores_by_class)
    print(f"Processed frames: {total_frames}")
    print(f"Average inference time: {avg_infer_ms:.2f} ms")
    print(f"Final mAP (proxy): {final_map * 100:.2f}%")
    # Optional: print per-class proxy AP
    if ap_by_class:
        print("Per-class AP (proxy):")
        for cid, ap in sorted(ap_by_class.items(), key=lambda x: -x[1]):
            print(f"  {get_label_name(labels, cid)} (id {cid}): {ap * 100:.2f}%")
    print(f"Saved output video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()