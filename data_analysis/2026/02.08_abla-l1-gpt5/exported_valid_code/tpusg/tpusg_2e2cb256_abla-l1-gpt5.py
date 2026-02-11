import os
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Configuration Parameters
model_path = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
label_path = "/home/mendel/tinyml_autopilot/models/labelmap.txt"
input_path = "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
output_path = "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4"
confidence_threshold = 0.5

def load_labels(path):
    labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    # Try to parse "id label" or "id: label" formats; fallback to index-based
    parsed_any = False
    for line in lines:
        parts = line.replace(":", " ").split()
        if parts and parts[0].isdigit():
            idx = int(parts[0])
            name = " ".join(parts[1:]) if len(parts) > 1 else str(idx)
            labels[idx] = name
            parsed_any = True
    if not parsed_any:
        for i, name in enumerate(lines):
            labels[i] = name
    return labels

def get_output_indices(interpreter):
    # Determine output tensor indices for boxes, classes, scores, and count
    details = interpreter.get_output_details()
    idx = {"boxes": None, "classes": None, "scores": None, "count": None}
    # First pass by name if available
    for d in details:
        name = d.get("name", "").lower()
        if "boxes" in name and idx["boxes"] is None:
            idx["boxes"] = d["index"]
        elif "classes" in name and idx["classes"] is None:
            idx["classes"] = d["index"]
        elif "scores" in name and idx["scores"] is None:
            idx["scores"] = d["index"]
        elif "num_detections" in name or "count" in name:
            idx["count"] = d["index"]

    # Fallback by shapes if any None
    if any(v is None for v in idx.values()):
        # Typical order for TFLite detection postprocess: boxes, classes, scores, count
        # Try to infer by shape
        for d in details:
            shp = d.get("shape", [])
            if len(shp) == 3 and shp[-1] == 4:
                idx["boxes"] = d["index"]
        # The remaining ones are 2D with shape [1, N] or 1D with shape [1]
        remaining = [d for d in details if d["index"] not in [idx["boxes"]]]
        # Try to detect count as shape [1]
        for d in remaining:
            shp = d.get("shape", [])
            if len(shp) == 1 and shp[0] == 1:
                idx["count"] = d["index"]
        # For classes and scores: look at dtype (classes usually float or int; scores float)
        for d in remaining:
            if d["index"] in [idx["count"]]:
                continue
            name = d.get("name", "").lower()
            if "classes" in name:
                idx["classes"] = d["index"]
            elif "scores" in name:
                idx["scores"] = d["index"]
        # If still missing, assign by remaining order
        if idx["classes"] is None or idx["scores"] is None:
            leftovers = [d for d in details if d["index"] not in [idx["boxes"], idx["count"]]]
            if len(leftovers) >= 2:
                # Heuristic: float dtype likely scores
                ld0, ld1 = leftovers[0], leftovers[1]
                if ld0["dtype"] == np.float32:
                    idx["scores"] = ld0["index"]
                    idx["classes"] = ld1["index"]
                else:
                    idx["classes"] = ld0["index"]
                    idx["scores"] = ld1["index"]
    return idx

def preprocess_frame(frame, input_details):
    h_in, w_in = input_details['shape'][1], input_details['shape'][2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (w_in, h_in))
    tensor = np.expand_dims(resized, axis=0)
    if input_details['dtype'] == np.float32:
        tensor = tensor.astype(np.float32) / 255.0
    else:
        tensor = tensor.astype(np.uint8)
    return tensor

def draw_detections(frame, detections, labels, conf_threshold):
    h, w = frame.shape[:2]
    for det in detections:
        ymin, xmin, ymax, xmax, score, cls_id = det
        if score < conf_threshold:
            continue
        x1 = int(max(0, xmin * w))
        y1 = int(max(0, ymin * h))
        x2 = int(min(w - 1, xmax * w))
        y2 = int(min(h - 1, ymax * h))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = labels.get(int(cls_id), str(int(cls_id)))
        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, thickness=-1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def update_map_stats(stats, detections, conf_threshold):
    # Proxy mAP stats: per-class precision approximation using thresholded vs total detections
    for det in detections:
        score = det[4]
        cls_id = int(det[5])
        if cls_id not in stats:
            stats[cls_id] = {"above": 0, "total": 0}
        if score >= conf_threshold:
            stats[cls_id]["above"] += 1
        stats[cls_id]["total"] += 1

def compute_proxy_map(stats):
    if not stats:
        return 0.0
    precisions = []
    for cls_id, v in stats.items():
        total = max(1, v["total"])
        precisions.append(v["above"] / total)
    if not precisions:
        return 0.0
    return float(np.mean(precisions))

def main():
    # Step 1: Setup - load interpreter with EdgeTPU, allocate tensors, load labels, open video
    labels = load_labels(label_path)

    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0')]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_indices = get_output_indices(interpreter)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video at {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not open output video for writing at {output_path}")
        cap.release()
        return

    # Stats for proxy mAP
    map_stats = {}
    frame_count = 0
    t0 = time.time()

    # Step 2-4: Process video
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1

        # Preprocessing
        input_tensor = preprocess_frame(frame, input_details)
        interpreter.set_tensor(input_details['index'], input_tensor)

        # Inference
        interpreter.invoke()

        # Output handling
        boxes = interpreter.get_tensor(output_indices["boxes"])[0]
        classes = interpreter.get_tensor(output_indices["classes"])[0]
        scores = interpreter.get_tensor(output_indices["scores"])[0]
        # num detections can be float or int
        if output_indices["count"] is not None:
            num = interpreter.get_tensor(output_indices["count"])
            try:
                num = int(num.flatten()[0])
            except Exception:
                num = boxes.shape[0]
        else:
            num = boxes.shape[0]

        detections = []
        for i in range(num):
            ymin, xmin, ymax, xmax = boxes[i]
            cls_id = int(classes[i]) if i < len(classes) else 0
            score = float(scores[i]) if i < len(scores) else 0.0
            # clamp coords to [0,1]
            ymin = float(np.clip(ymin, 0.0, 1.0))
            xmin = float(np.clip(xmin, 0.0, 1.0))
            ymax = float(np.clip(ymax, 0.0, 1.0))
            xmax = float(np.clip(xmax, 0.0, 1.0))
            detections.append((ymin, xmin, ymax, xmax, score, cls_id))

        # Draw detections above threshold
        draw_detections(frame, detections, labels, confidence_threshold)

        # Update and draw proxy mAP
        update_map_stats(map_stats, detections, confidence_threshold)
        current_map = compute_proxy_map(map_stats)
        map_text = f"mAP: {current_map:.3f}"
        (tw, th), baseline = cv2.getTextSize(map_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 6, 10 + th + baseline + 6), (255, 255, 255), thickness=-1)
        cv2.putText(frame, map_text, (13, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Write frame
        writer.write(frame)

    # Cleanup
    writer.release()
    cap.release()
    elapsed = time.time() - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

    final_map = compute_proxy_map(map_stats)
    print("Processing complete.")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Proxy mAP over video: {final_map:.3f}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()