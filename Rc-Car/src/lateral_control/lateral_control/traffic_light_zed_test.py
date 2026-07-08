import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorflow_hub as hub
except ImportError:
    hub = None

try:
    import tf_keras as keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        keras = None

try:
    import pyzed.sl as sl
except ImportError:
    sl = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_YOLO_MODEL_PATH = (
    "/home/wsadmin/Desktop/Rc-Car-Chimera/Rc-Car-Chimera/Rc-Car/src/model/best.pt"
)

MODEL_FILE_NAMES = [
    "20260701-08091782893391-all-images-mobilenetv2-Adam.h5",
]

LABELS = ["back", "green", "red", "yellow"]

DISPLAY_COLORS = {
    "back": (160, 160, 160),
    "green": (0, 200, 0),
    "red": (0, 0, 255),
    "yellow": (0, 220, 255),
    "unknown": (160, 160, 160),
}

ACTION_BY_LABEL = {
    "red": "STOP",
    "yellow": "SLOW",
    "green": "GO",
    "back": "NO_TRAFFIC_LIGHT",
    "unknown": "NO_TRAFFIC_LIGHT",
}

COLOR_RANGES = {
    "red": [
        ((0, 80, 110), (12, 255, 255)),
        ((168, 80, 110), (180, 255, 255)),
    ],
    "yellow": [
        ((15, 60, 110), (38, 255, 255)),
    ],
    "green": [
        ((38, 35, 60), (90, 255, 255)),
        ((90, 35, 70), (98, 255, 255)),
    ],
}

COLOR_OVERRIDE_MIN_CONFIDENCE = 0.18
DISTANCE_HISTORY = []
DISTANCE_JUMP_COUNT = 0
MAX_DISTANCE_HISTORY = 9
MAX_SMOOTH_DISTANCE_JUMP = 0.45
MIN_JUMP_FRAMES_TO_RESET = 3
DEPTH_BBOX_INSET_RATIO = 0.30
MAX_MISSED_DETECTION_FRAMES = 12

def action_for_label(label):
    return ACTION_BY_LABEL.get(label, ACTION_BY_LABEL["unknown"])


def proximity_from_distance(distance_m, near_threshold, far_threshold):
    if distance_m is None:
        return "unknown"

    if distance_m < near_threshold:
        return "aproape"

    if distance_m > far_threshold:
        return "departe"

    return "mediu"

def proximity_from_bbox_height(bbox_height, near_height, far_height):
    if bbox_height is None:
        return "unknown"

    if bbox_height >= near_height:
        return "aproape"

    if bbox_height <= far_height:
        return "departe"

    return "mediu"

def estimate_distance_from_bbox(bbox, bbox_distance_k, bbox_distance_offset):
    x1, y1, x2, y2 = bbox

    box_h = y2 - y1

    if box_h <= 0:
        return None

    distance_m = bbox_distance_k / box_h + bbox_distance_offset

    return float(max(0.20, distance_m))

def estimate_distance_from_depth(depth_frame, bbox):
    if depth_frame is None:
        return None

    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None

    # Nu folosim tot bbox-ul pentru depth.
    # Taiem 30% din fiecare margine si pastram doar zona centrala a semaforului.
    inset_x = int(box_w * DEPTH_BBOX_INSET_RATIO)
    inset_y = int(box_h * DEPTH_BBOX_INSET_RATIO)

    cx1 = x1 + inset_x
    cx2 = x2 - inset_x
    cy1 = y1 + inset_y
    cy2 = y2 - inset_y

    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    # Bbox-ul vine din imaginea LEFT ZED completa, aceleasi coordonate ca depth_frame.
    # Crop-ul RGB pentru culoare nu schimba coordonatele folosite aici.
    depth_crop = depth_frame[cy1:cy2, cx1:cx2]

    if depth_crop.size == 0:
        return None

    valid_depth = depth_crop[
        np.isfinite(depth_crop)
        & (depth_crop > 0.15)
        & (depth_crop < 5.0)
    ]

    if valid_depth.size < 20:
        return None

    # Eliminăm extremele, pentru că ZED mai dă valori aiurea.
    p20 = np.percentile(valid_depth, 20)
    p80 = np.percentile(valid_depth, 80)

    filtered_depth = valid_depth[
        (valid_depth >= p20)
        & (valid_depth <= p80)
    ]

    if filtered_depth.size < 10:
        return None

    return float(np.median(filtered_depth))

def format_distance(distance_m):
    if distance_m is None:
        return "DIST: ?"

    return f"DIST: {distance_m:.2f}m"

def smooth_distance(distance_m):
    global DISTANCE_HISTORY, DISTANCE_JUMP_COUNT

    if distance_m is None:
        DISTANCE_HISTORY = []
        DISTANCE_JUMP_COUNT = 0
        return None

    if len(DISTANCE_HISTORY) > 0:
        current_median = float(np.median(DISTANCE_HISTORY))

        if abs(distance_m - current_median) > MAX_SMOOTH_DISTANCE_JUMP:
            DISTANCE_JUMP_COUNT += 1

            if DISTANCE_JUMP_COUNT < MIN_JUMP_FRAMES_TO_RESET:
                return current_median

            DISTANCE_HISTORY = []
            DISTANCE_JUMP_COUNT = 0
        else:
            DISTANCE_JUMP_COUNT = 0

    DISTANCE_HISTORY.append(distance_m)

    if len(DISTANCE_HISTORY) > MAX_DISTANCE_HISTORY:
        DISTANCE_HISTORY = DISTANCE_HISTORY[-MAX_DISTANCE_HISTORY:]

    return float(np.median(DISTANCE_HISTORY))


def infer_color_from_crop(crop, color_ratio_threshold):
    if crop is None or crop.size == 0:
        return "unknown", 0.0

    # Mărim crop-ul pentru semafoare mici / la distanță.
    crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]

    # Scoatem marginile ca să ignorăm rama semaforului / fundalul.
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.05)

    hsv_inner = hsv[
        margin_y : h - margin_y if h - margin_y > margin_y else h,
        margin_x : w - margin_x if w - margin_x > margin_x else w,
    ]

    if hsv_inner.size == 0:
        hsv_inner = hsv

    saturation = hsv_inner[:, :, 1]
    value = hsv_inner[:, :, 2]

    # Pixeli suficient de colorați și luminoși.
    bright_mask = ((saturation > 45) & (value > 70)).astype(np.uint8) * 255

    active_pixels = max(1, cv2.countNonZero(bright_mask))

    scores = {}

    for label, ranges in COLOR_RANGES.items():
        color_mask = np.zeros(hsv_inner.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            current_mask = cv2.inRange(
                hsv_inner,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            color_mask = cv2.bitwise_or(color_mask, current_mask)

        color_mask = cv2.bitwise_and(color_mask, bright_mask)

        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        color_pixels = cv2.countNonZero(color_mask)
        score = color_pixels / float(active_pixels)

        scores[label] = score

    red_score = scores.get("red", 0.0)
    yellow_score = scores.get("yellow", 0.0)
    green_score = scores.get("green", 0.0)

    if red_score >= 0.08 and red_score >= green_score * 0.80:
        return "red", red_score

    if green_score >= 0.12 and green_score > red_score * 1.20:
        return "green", green_score

    if (
        yellow_score >= 0.08
        and yellow_score > red_score * 0.80
        and yellow_score > green_score * 0.80
    ):
        return "yellow", yellow_score

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score < 0.07:
        return "unknown", best_score

    return best_label, best_score


def normalize_yolo_label(class_name):
    class_name = class_name.lower().strip()

    if "red" in class_name:
        return "red"
    if "yellow" in class_name:
        return "yellow"
    if "green" in class_name:
        return "green"

    return "unknown"


def resolve_model_path(configured_path):
    if configured_path:
        model_path = Path(configured_path).expanduser()

        if model_path.exists():
            return str(model_path)

        raise FileNotFoundError(f"Model not found: {model_path}")

    project_root = Path(__file__).resolve().parents[3]

    search_dirs = [
        Path("/home/wsadmin/Desktop/Rc-Car-Chimera/Rc-Car-Chimera/Rc-Car/src/model"),
        project_root / "src" / "model",
        Path.cwd() / "src" / "model",
        Path(__file__).resolve().parents[2] / "model",
    ]

    for search_dir in search_dirs:
        for model_file_name in MODEL_FILE_NAMES:
            candidate = search_dir / model_file_name

            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "Traffic light Keras model was not found. "
        "Use --model-path with the full path of your .h5 model."
    )


def resolve_yolo_model_path(configured_path):
    model_path = Path(configured_path).expanduser()

    if model_path.exists() and model_path.is_file():
        return str(model_path)

    raise FileNotFoundError(
        "YOLO model was not found. Expected file:\n"
        f"{model_path}\n\n"
        "Important: use the .pt file directly. Do not extract/unzip best.pt."
    )


def preprocess(frame, input_size):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32)
    return np.expand_dims(image, axis=0)


def predict_label(model, frame, threshold, input_size):
    predictions = model.predict(preprocess(frame, input_size), verbose=0)
    probabilities = predictions[0]

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    if class_index >= len(LABELS):
        return "unknown", confidence, probabilities, class_index

    label = LABELS[class_index]

    if confidence < threshold:
        return "unknown", confidence, probabilities, class_index

    return label, confidence, probabilities, class_index


def make_one_hot(class_index, size):
    return [1 if i == class_index else 0 for i in range(size)]


def print_prediction(label, confidence, probabilities, class_index):
    one_hot = make_one_hot(class_index, len(LABELS))

    print("\n==============================")
    print("SCAN RESULT")
    print("==============================")
    print(f"Detected label: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class index: {class_index}")
    print(f"One-hot: {one_hot}")

    print("\nProbabilities:")
    for i, label_name in enumerate(LABELS):
        value = float(probabilities[i]) if i < len(probabilities) else 0.0
        print(f"  {i} - {label_name}: {value:.4f}")

    print("==============================\n")


def crop_center(frame, scale):
    h, w = frame.shape[:2]

    crop_w = int(w * scale)
    crop_h = int(h * scale)

    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    crop = frame[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)


def get_roi(frame, args):
    if args.no_crop or args.roi_scale >= 1.0:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h)

    if args.roi is not None:
        x, y, w, h = args.roi

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    return crop_center(frame, args.roi_scale)


def draw_preview(frame, args):
    preview = frame.copy()

    _, roi_box = get_roi(preview, args)
    x1, y1, x2, y2 = roi_box

    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 90), (25, 25, 25), -1)

    cv2.putText(
        preview,
        "Press S to scan | Press Q to quit",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    return preview


def draw_result(frame, roi_box, label, confidence, probabilities):
    x1, y1, x2, y2 = roi_box

    color = DISPLAY_COLORS.get(label, DISPLAY_COLORS["unknown"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (25, 25, 25), -1)

    cv2.putText(
        frame,
        f"{label.upper()}  {confidence:.2f}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        color,
        3,
        cv2.LINE_AA,
    )

    y = 130

    for index, label_name in enumerate(LABELS):
        value = float(probabilities[index]) if index < len(probabilities) else 0.0

        cv2.putText(
            frame,
            f"{label_name}: {value:.2f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            DISPLAY_COLORS.get(label_name, DISPLAY_COLORS["unknown"]),
            2,
            cv2.LINE_AA,
        )

        y += 30


def class_name_from_yolo(model, class_index):
    names = model.names

    if isinstance(names, dict):
        return str(names.get(class_index, class_index)).lower()

    return str(names[class_index]).lower()


def detect_yolo_traffic_light(model, frame, args, depth_frame=None):
    results = model.predict(
    frame,
    conf=args.yolo_conf,
    imgsz=args.yolo_imgsz,
    device="cpu",
    verbose=False,
)

    if not results:
        return None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    valid_classes = {
        item.strip().lower()
        for item in args.yolo_valid_classes.split(",")
        if item.strip()
    }

    best_detection = None
    frame_h, frame_w = frame.shape[:2]

    for box in result.boxes:
        class_index = int(box.cls[0])
        class_name = class_name_from_yolo(model, class_index)

        is_valid_class = (
            class_name in valid_classes
            or "traffic" in class_name
            or "red" in class_name
            or "yellow" in class_name
            or "green" in class_name
        )

        if not is_valid_class:
            continue

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue
        if y1 <= 5 or x1 <= 5 or x2 >= frame_w - 5:
            continue

        bbox = (x1, y1, x2, y2)
        crop = frame[y1:y2, x1:x2]

        yolo_label = normalize_yolo_label(class_name)
        crop_label, crop_confidence = infer_color_from_crop(
            crop,
            args.color_ratio_threshold,
        )

        label = yolo_label
        color_confidence = crop_confidence

        if crop_label != "unknown":
            strong_crop_color = crop_confidence >= max(
                args.color_ratio_threshold,
                COLOR_OVERRIDE_MIN_CONFIDENCE,
            )

            if yolo_label == "unknown" or (
                crop_label != yolo_label and strong_crop_color
            ):
                label = crop_label
            elif crop_confidence >= args.color_ratio_threshold:
                label = crop_label

        if label == "unknown":
            color_confidence = confidence

        if label == "unknown":
            continue

        bbox_height = y2 - y1

        bbox_distance_m = estimate_distance_from_bbox(
            bbox,
            args.bbox_distance_k,
            args.bbox_distance_offset,
        )

        zed_distance_m = estimate_distance_from_depth(
            depth_frame,
            bbox,
        )

        distance_source = "unknown"

        if args.distance_method == "bbox":
            distance_m = bbox_distance_m
            distance_source = "bbox"
        elif args.distance_method == "zed":
            distance_m = zed_distance_m
            distance_source = "zed"
        else:
            zed_matches_bbox = (
                zed_distance_m is not None
                and bbox_distance_m is not None
                and 0.25 <= zed_distance_m <= 5.0
                and abs(zed_distance_m - bbox_distance_m) <= max(
                    0.20,
                    bbox_distance_m * 0.25,
                )
            )

            if zed_matches_bbox:
                distance_m = zed_distance_m
                distance_source = "zed"
            else:
                distance_m = bbox_distance_m
                distance_source = "bbox"

        raw_distance_m = distance_m

        if distance_m is not None:
            distance_m = smooth_distance(distance_m)

        if args.proximity_method == "bbox_height":
            proximity = proximity_from_bbox_height(
                bbox_height,
                args.near_bbox_height,
                args.far_bbox_height,
            )
        else:
            proximity = proximity_from_distance(
                distance_m,
                args.near_distance,
                args.far_distance,
            )

        bbox_area = (x2 - x1) * (y2 - y1)
        normalized_area = bbox_area / float(frame_w * frame_h)

        score = confidence + color_confidence + normalized_area

        if best_detection is None or score > best_detection["score"]:
            best_detection = {
                "label": label,
                "action": action_for_label(label),
                "confidence": confidence,
                "color_confidence": color_confidence,
                "bbox": bbox,
                "bbox_height": bbox_height,
                "class_name": class_name,
                "yolo_label": yolo_label,
                "crop_label": crop_label,
                "score": score,
                "distance_m": distance_m,
                "raw_distance_m": raw_distance_m,
                "bbox_distance_m": bbox_distance_m,
                "zed_distance_m": zed_distance_m,
                "distance_source": distance_source,
                "proximity": proximity,
            }

    return best_detection

def draw_yolo_result(frame, detection):
    display = frame.copy()

    if detection is None:
        label = "unknown"
        color = DISPLAY_COLORS["unknown"]

        cv2.rectangle(display, (0, 0), (display.shape[1], 110), (25, 25, 25), -1)

        cv2.putText(
            display,
            "SEARCHING TRAFFIC LIGHT",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "DIST: ?  UNKNOWN",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        return display

    label = detection["label"]
    confidence = detection["confidence"]
    action = detection["action"]
    color = DISPLAY_COLORS.get(label, DISPLAY_COLORS["unknown"])
    x1, y1, x2, y2 = detection["bbox"]

    distance_m = detection.get("distance_m")
    proximity = detection.get("proximity", "unknown")
    distance_text = format_distance(distance_m)

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(display, (0, 0), (display.shape[1], 110), (25, 25, 25), -1)

    cv2.putText(
        display,
        f"{label.upper()}  {confidence:.2f}  {action}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"{distance_text}  {proximity.upper()}",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return display


def print_yolo_detection(detection, previous_state):
    if detection is None:
        label = "unknown"
        confidence = 0.0
        action = action_for_label(label)
        bbox = "none"
        bbox_height = "none"
        class_name = "none"
        color_confidence = 0.0
        yolo_label = "unknown"
        crop_label = "unknown"
        distance_m = None
        raw_distance_m = None
        bbox_distance_m = None
        zed_distance_m = None
        distance_source = "unknown"
        proximity = "unknown"
    else:
        label = detection["label"]
        confidence = detection["confidence"]
        action = detection["action"]
        bbox = detection["bbox"]
        bbox_height = detection.get("bbox_height", "unknown")
        class_name = detection["class_name"]
        color_confidence = detection["color_confidence"]
        yolo_label = detection.get("yolo_label", "unknown")
        crop_label = detection.get("crop_label", "unknown")
        distance_m = detection.get("distance_m")
        raw_distance_m = detection.get("raw_distance_m")
        bbox_distance_m = detection.get("bbox_distance_m")
        zed_distance_m = detection.get("zed_distance_m")
        distance_source = detection.get("distance_source", "unknown")
        proximity = detection.get("proximity", "unknown")

    current_state = f"{label}:{proximity}:{distance_m}"

    if current_state == previous_state:
        return previous_state

    print("\n================ YOLO TRAFFIC LIGHT LIVE ================")
    print(f"state: {label}")
    print(f"action: {action}")
    print(f"yolo_class: {class_name}")
    print(f"yolo_label: {yolo_label}")
    print(f"crop_label: {crop_label}")
    print(f"yolo_confidence: {confidence:.2f}")
    print(f"color_confidence: {color_confidence:.3f}")

    if raw_distance_m is None:
        print("raw_distance: unknown")
    else:
        print(f"raw_distance: {raw_distance_m:.2f} m")

    if bbox_distance_m is None:
        print("bbox_distance: unknown")
    else:
        print(f"bbox_distance: {bbox_distance_m:.2f} m")

    if zed_distance_m is None:
        print("zed_distance: unknown")
    else:
        print(f"zed_distance: {zed_distance_m:.2f} m")

    if distance_m is None:
        print("distance: unknown")
    else:
        print(f"distance: {distance_m:.2f} m")

    print(f"distance_source: {distance_source}")
    print(f"proximity: {proximity}")
    print(f"bbox: {bbox}")
    print(f"bbox_height: {bbox_height}")
    print("=========================================================\n")

    return current_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test traffic light detection using ZED camera."
    )

    parser.add_argument(
        "--detector",
        choices=("yolo", "keras"),
        default="yolo",
        help="Detector backend. Default uses YOLO live detection.",
    )

    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL_PATH,
        help="YOLO model path. Default uses src/model/best.pt.",
    )

    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.20,
        help="Minimum YOLO confidence.",
    )

    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=960,
        help="YOLO inference image size. Use 640 for speed, 960/1280 for distant traffic lights.",
    )

    parser.add_argument(
        "--yolo-valid-classes",
        default=(
            "traffic light,traffic_light,"
            "red,yellow,green,"
            "red_light,yellow_light,green_light,"
            "red_traffic_light,yellow_traffic_light,green_traffic_light"
        ),
        help="Comma-separated YOLO class names accepted as traffic lights.",
    )

    parser.add_argument(
        "--color-ratio-threshold",
        type=float,
        default=0.015,
        help="Minimum color ratio inside YOLO bbox when YOLO class is only traffic light.",
    )

    parser.add_argument(
        "--near-distance",
        type=float,
        default=0.6,
        help="Distance in meters below which the traffic light is considered close.",
    )

    parser.add_argument(
        "--far-distance",
        type=float,
        default=1.2,
        help="Distance in meters above which the traffic light is considered far.",
    )

    parser.add_argument(
        "--proximity-method",
        choices=("bbox_height", "distance"),
        default="bbox_height",
        help="How to classify close/medium/far. bbox_height is more stable for this test.",
    )

    parser.add_argument(
        "--near-bbox-height",
        type=int,
        default=130,
        help="BBox height in pixels above which the traffic light is considered close.",
    )

    parser.add_argument(
        "--far-bbox-height",
        type=int,
        default=85,
        help="BBox height in pixels below which the traffic light is considered far.",
    )

    parser.add_argument(
        "--model-path",
        default="",
        help="Full path to the .h5 model.",
    )

    parser.add_argument(
        "--image",
        default="",
        help="Path to a static image instead of ZED camera.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum confidence required for a valid Keras class.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Keras model input width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Keras model input height.",
    )

    parser.add_argument(
        "--roi-scale",
        type=float,
        default=1.0,
        help="Center crop size for Keras mode. Default 1.0 uses full frame.",
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Use full frame instead of center crop in Keras mode.",
    )

    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Manual crop area for Keras mode: x y width height.",
    )
    parser.add_argument(
        "--distance-method",
        choices=("bbox", "zed", "hybrid"),
        default="bbox",
        help="Distance method: bbox is calibrated from YOLO bbox height.",
    )

    parser.add_argument(
        "--bbox-distance-k",
        type=float,
        default=182.0,
        help="Scale for bbox distance: distance = scale / bbox_height_px + offset.",
    )

    parser.add_argument(
        "--bbox-distance-offset",
        type=float,
        default=-0.89,
        help="Offset for bbox distance: distance = scale / bbox_height_px + offset.",
    )
    return parser.parse_args()


def open_zed_camera():
    if sl is None:
        raise RuntimeError(
            "ZED SDK Python API is not installed. Install pyzed first."
        )

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Avem nevoie de depth pentru distanța reală ZED.
    # Nu rula cu CUDA_VISIBLE_DEVICES="", pentru că ZED are nevoie de GPU.
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    return zed

def grab_zed_frame(zed):
    runtime_params = sl.RuntimeParameters()
    zed_image = sl.Mat()
    zed_depth = sl.Mat()

    status = zed.grab(runtime_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not grab frame from ZED: {status}")

    zed.retrieve_image(zed_image, sl.VIEW.LEFT)
    zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)

    frame_bgra = zed_image.get_data()
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    depth_frame = zed_depth.get_data()

    return frame_bgr, depth_frame


def load_image(image_path):
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return frame


def scan_frame(model, frame, args):
    input_size = (args.width, args.height)

    roi_frame, roi_box = get_roi(frame, args)

    if roi_frame.size == 0:
        raise RuntimeError("ROI crop is empty. Check --roi values.")

    label, confidence, probabilities, class_index = predict_label(
        model,
        roi_frame,
        args.threshold,
        input_size,
    )

    print_prediction(label, confidence, probabilities, class_index)

    cv2.imwrite("zed_scan_full_frame.jpg", frame)
    cv2.imwrite("zed_scan_roi_used_by_model.jpg", roi_frame)

    print("Saved full frame: zed_scan_full_frame.jpg")
    print("Saved ROI used by model: zed_scan_roi_used_by_model.jpg")

    result_frame = frame.copy()
    draw_result(result_frame, roi_box, label, confidence, probabilities)

    return result_frame


def load_detector(args):
    if args.detector == "yolo":
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install it with:\n"
                "python3 -m pip install ultralytics"
            )

        yolo_model_path = resolve_yolo_model_path(args.yolo_model)

        print(f"Using YOLO model: {yolo_model_path}")
        print(f"YOLO confidence: {args.yolo_conf}")
        print(f"YOLO image size: {args.yolo_imgsz}")
        print(f"Near distance: {args.near_distance} m")
        print(f"Far distance: {args.far_distance} m")
        print("YOLO live mode: no S key needed. Press ESC to quit.")

        return YOLO(yolo_model_path)

    if keras is None:
        raise RuntimeError(
            "TensorFlow/Keras is not installed. Install it only if you want Keras mode."
        )

    if hub is None:
        raise RuntimeError(
            "tensorflow_hub is not installed. Install it only if you want Keras mode."
        )

    model_path = resolve_model_path(args.model_path)

    print(f"Using Keras model: {model_path}")
    print(f"Labels order: {LABELS}")
    print("IMPORTANT: Labels order must match train_ds.class_names from Colab.")

    return keras.models.load_model(
        model_path,
        custom_objects={"KerasLayer": hub.KerasLayer},
    )


def main():
    args = parse_args()
    model = load_detector(args)

    # Static image mode.
    if args.image:
        print(f"Using static image: {args.image}")

        frame = load_image(args.image)

        if args.detector == "yolo":
            detection = detect_yolo_traffic_light(
                model,
                frame,
                args,
                depth_frame=None,
            )
            print_yolo_detection(detection, previous_state=None)
            result_frame = draw_yolo_result(frame, detection)
        else:
            result_frame = scan_frame(model, frame, args)

        cv2.imshow("Traffic Light Image Scan Result", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ZED camera mode.
    print("Opening ZED camera...")
    zed = open_zed_camera()
    print("ZED camera opened.")

    try:
        previous_state = None
        last_yolo_detection = None
        missed_detection_frames = 0

        while True:
            frame, depth_frame = grab_zed_frame(zed)

            if args.detector == "yolo":
                detection = detect_yolo_traffic_light(
                    model,
                    frame,
                    args,
                    depth_frame=depth_frame,
                )   

                if detection is None:
                    missed_detection_frames += 1

                    if (
                        last_yolo_detection is not None
                        and missed_detection_frames <= MAX_MISSED_DETECTION_FRAMES
                    ):
                        detection = last_yolo_detection.copy()
                        detection["class_name"] = "last_valid_detection"
                else:
                    last_yolo_detection = detection.copy()
                    missed_detection_frames = 0

                previous_state = print_yolo_detection(detection, previous_state)
                display_frame = draw_yolo_result(frame, detection)
                window_name = "Traffic Light YOLO Live"
            else:
                result_frame = scan_frame(model, frame, args)
                display_frame = result_frame
                window_name = "Traffic Light Keras Live"

            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) == 27:
                break

    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
