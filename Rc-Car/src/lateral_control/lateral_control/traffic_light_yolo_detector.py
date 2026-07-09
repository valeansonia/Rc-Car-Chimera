from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32MultiArray, String

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


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


DISPLAY_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 220, 255),
    "green": (0, 200, 0),
    "back": (160, 160, 160),
    "unknown": (160, 160, 160),
}


def action_for_state(state):
    if state == "red":
        return "STOP"
    if state == "yellow":
        return "SLOW"
    if state == "green":
        return "GO"
    return "NO_TRAFFIC_LIGHT"


class TrafficLightYoloDetector(Node):
    def __init__(self):
        super().__init__("traffic_light_yolo_detector")

        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Install it with: "
                "python3 -m pip install ultralytics"
            )

        self.declare_parameter("image_topic", "/ZEDcam/image_raw")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("confidence_threshold", 0.45)
        self.declare_parameter("color_ratio_threshold", 0.015)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter(
            "valid_classes",
            "traffic light,traffic_light,red,yellow,green,red_light,yellow_light,green_light",
        )

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = str(Path(self.get_parameter("model_path").value).expanduser())
        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self.color_ratio_threshold = float(
            self.get_parameter("color_ratio_threshold").value
        )
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.valid_classes = {
            item.strip().lower()
            for item in self.get_parameter("valid_classes").value.split(",")
            if item.strip()
        }

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.names = self.model.names

        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.state_publisher = self.create_publisher(String, "/traffic_light/state", 10)
        self.action_publisher = self.create_publisher(String, "/traffic_light/action", 10)
        self.confidence_publisher = self.create_publisher(
            Float32,
            "/traffic_light/confidence",
            10,
        )
        self.stop_publisher = self.create_publisher(Bool, "/traffic_light/stop", 10)
        self.bbox_publisher = self.create_publisher(
            Int32MultiArray,
            "/traffic_light/bbox",
            10,
        )

        self.debug_image_publisher = None
        if self.publish_debug_image:
            self.debug_image_publisher = self.create_publisher(
                Image,
                "/traffic_light/debug_image",
                10,
            )

        self.get_logger().info(
            "Traffic light YOLO detector started on topic "
            f"{self.image_topic} using model {self.model_path}"
        )

    def class_name(self, class_index):
        if isinstance(self.names, dict):
            return str(self.names.get(class_index, class_index)).lower()
        return str(self.names[class_index]).lower()

    def normalize_yolo_label(self, class_name):
        class_name = class_name.lower()

        if "red" in class_name:
            return "red"
        if "yellow" in class_name:
            return "yellow"
        if "green" in class_name:
            return "green"

        return "unknown"

    def infer_color_from_crop(self, crop):
        if crop is None or crop.size == 0:
            return "unknown", 0.0

        crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

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
        active_mask = ((saturation > 45) & (value > 70)).astype(np.uint8) * 255
        active_pixels = max(1, cv2.countNonZero(active_mask))
        scores = {}

        for label, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv_inner.shape[:2], dtype=np.uint8)

            for lower, upper in ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(
                        hsv_inner,
                        np.array(lower, dtype=np.uint8),
                        np.array(upper, dtype=np.uint8),
                    ),
                )

            mask = cv2.bitwise_and(mask, active_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            scores[label] = float(cv2.countNonZero(mask)) / float(active_pixels)

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

        if best_score < self.color_ratio_threshold:
            return "unknown", best_score

        return best_label, best_score

    def detect_traffic_light(self, frame):
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        if not results:
            return None

        best_detection = None

        for box in results[0].boxes:
            class_index = int(box.cls[0])
            class_name = self.class_name(class_index)

            if class_name not in self.valid_classes:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            crop = frame[y1:y2, x1:x2]

            yolo_label = self.normalize_yolo_label(class_name)
            crop_label, crop_confidence = self.infer_color_from_crop(crop)

            label = yolo_label
            color_confidence = crop_confidence

            if crop_label != "unknown":
                strong_crop_color = crop_confidence >= max(
                    self.color_ratio_threshold,
                    COLOR_OVERRIDE_MIN_CONFIDENCE,
                )

                if yolo_label == "unknown" or (
                    crop_label != yolo_label and strong_crop_color
                ):
                    label = crop_label
                elif crop_confidence >= self.color_ratio_threshold:
                    label = crop_label

            if label == "unknown":
                color_confidence = confidence

            if label == "unknown":
                continue

            score = confidence + color_confidence

            if best_detection is None or score > best_detection["score"]:
                best_detection = {
                    "label": label,
                    "action": action_for_state(label),
                    "confidence": confidence,
                    "color_confidence": color_confidence,
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "score": score,
                }

        return best_detection

    def publish_detection(self, detection):
        if detection is None:
            state = "back"
            action = action_for_state(state)
            confidence = 0.0
            bbox = [0, 0, 0, 0]
        else:
            state = detection["label"]
            action = detection["action"]
            confidence = detection["confidence"]
            bbox = list(detection["bbox"])

        self.state_publisher.publish(String(data=state))
        self.action_publisher.publish(String(data=action))
        self.confidence_publisher.publish(Float32(data=confidence))
        self.stop_publisher.publish(Bool(data=(state == "red")))
        self.bbox_publisher.publish(Int32MultiArray(data=bbox))

        print(
            "LOGGING: YOLO traffic light -> "
            f"state={state}, action={action}, confidence={confidence:.2f}, "
            f"bbox={bbox}"
        )

    def draw_detection(self, frame, detection):
        display = frame.copy()

        if detection is None:
            text = "NO TRAFFIC LIGHT"
            color = DISPLAY_COLORS["unknown"]
        else:
            state = detection["label"]
            action = detection["action"]
            confidence = detection["confidence"]
            x1, y1, x2, y2 = detection["bbox"]
            color = DISPLAY_COLORS.get(state, DISPLAY_COLORS["unknown"])

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            text = f"{state.upper()} {confidence:.2f} {action}"

        cv2.rectangle(display, (0, 0), (display.shape[1], 76), (25, 25, 25), -1)
        cv2.putText(
            display,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
            cv2.LINE_AA,
        )

        return display

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            detection = self.detect_traffic_light(frame)
            self.publish_detection(detection)

            if self.debug_image_publisher is not None:
                debug_image = self.draw_detection(frame, detection)
                self.debug_image_publisher.publish(
                    self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                )
        except CvBridgeError as error:
            self.get_logger().error(f"Could not convert ROS image: {error}")
        except Exception as error:
            self.get_logger().error(f"YOLO traffic light detection failed: {error}")


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightYoloDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
