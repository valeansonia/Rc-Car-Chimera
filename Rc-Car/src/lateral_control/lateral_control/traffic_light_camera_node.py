from pathlib import Path
from types import SimpleNamespace

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32MultiArray, String

from . import traffic_light_zed_test as detector


class TrafficLightCameraNode(Node):
    """Ruleaza detectorul de semafor pe imaginile publicate deja de ROS 2."""

    def __init__(self):
        super().__init__("traffic_light_camera_node")

        self.declare_parameter("image_topic", "/ZEDcam/image_raw")
        self.declare_parameter("model_path", "src/model/bestSem.pt")
        self.declare_parameter("yolo_conf", 0.08)
        self.declare_parameter("yolo_imgsz", 1280)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("show_window", False)
        self.declare_parameter("use_left_zed_image", True)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        self.show_window = bool(self.get_parameter("show_window").value)
        self.use_left_zed_image = bool(
            self.get_parameter("use_left_zed_image").value
        )

        model_path = self.resolve_model_path(
            str(self.get_parameter("model_path").value)
        )
        self.args = self.make_detector_args(model_path)
        self.model = detector.load_detector(self.args)
        self.bridge = CvBridge()
        self.previous_state = None
        self.last_detection = None
        self.missed_detection_frames = 0

        detector.reset_color_filter()
        detector.reset_distance_filter()

        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.state_publisher = self.create_publisher(
            String, "/traffic_light/state", 10
        )
        self.action_publisher = self.create_publisher(
            String, "/traffic_light/action", 10
        )
        self.confidence_publisher = self.create_publisher(
            Float32, "/traffic_light/confidence", 10
        )
        self.distance_publisher = self.create_publisher(
            Float32, "/traffic_light/distance", 10
        )
        self.stop_publisher = self.create_publisher(
            Bool, "/traffic_light/stop", 10
        )
        self.bbox_publisher = self.create_publisher(
            Int32MultiArray, "/traffic_light/bbox", 10
        )
        self.debug_image_publisher = None
        if self.publish_debug_image:
            self.debug_image_publisher = self.create_publisher(
                Image, "/traffic_light/debug_image", 10
            )

        self.get_logger().info(
            "Traffic-light camera node started: "
            f"image_topic={self.image_topic}, model={model_path}. "
            "Shared-camera mode uses bbox distance (no ZED depth map)."
        )

    def resolve_model_path(self, configured_path):
        configured = Path(configured_path).expanduser()
        candidates = [configured]

        if not configured.is_absolute():
            candidates.extend(
                [
                    Path.cwd() / configured,
                    Path(__file__).resolve().parents[3] / configured,
                ]
            )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())

        checked = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "YOLO traffic-light model was not found. Checked:\n" + checked
        )

    def make_detector_args(self, model_path):
        return SimpleNamespace(
            detector="yolo",
            yolo_model=model_path,
            yolo_conf=float(self.get_parameter("yolo_conf").value),
            yolo_imgsz=int(self.get_parameter("yolo_imgsz").value),
            yolo_valid_classes=(
                "traffic light,traffic_light,red,yellow,green,"
                "red_light,yellow_light,green_light,"
                "red_traffic_light,yellow_traffic_light,green_traffic_light"
            ),
            color_ratio_threshold=0.015,
            distance_method="bbox",
            proximity_method="bbox_height",
            near_distance=0.60,
            far_distance=1.20,
            near_bbox_height=130,
            far_bbox_height=85,
            bbox_distance_k=182.0,
            bbox_distance_offset=-0.89,
        )

    def publish_detection(self, detection):
        if detection is None:
            state = "unknown"
            action = detector.action_for_label(state)
            confidence = 0.0
            distance_m = float("nan")
            bbox = []
        else:
            state = detection["label"]
            action = detection["action"]
            confidence = float(detection["confidence"])
            measured_distance = detection.get("distance_m")
            distance_m = (
                float(measured_distance)
                if measured_distance is not None
                else float("nan")
            )
            bbox = list(detection.get("bbox", ()))

        self.state_publisher.publish(String(data=state))
        self.action_publisher.publish(String(data=action))
        self.confidence_publisher.publish(Float32(data=confidence))
        self.distance_publisher.publish(Float32(data=distance_m))
        self.stop_publisher.publish(Bool(data=state == "red"))
        self.bbox_publisher.publish(Int32MultiArray(data=bbox))

    def image_callback(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            detection_frame = frame
            if self.use_left_zed_image and frame.shape[1] >= 1000:
                # usb_cam publica ZED-ul ca imagine stereo side-by-side.
                # Lane detection foloseste jumatatea stanga; folosim aceeasi
                # imagine ca sa nu detectam semaforul de doua ori.
                detection_frame = frame[:, : frame.shape[1] // 2]

            # Parametrii testului direct au fost calibrati la HD720. Camera
            # ROS pentru lane detection ruleaza la 376p, deci scalarea dupa
            # inaltime pastreaza aceleasi praguri fizice pentru bbox.
            resolution_scale = detection_frame.shape[0] / 720.0
            self.args.bbox_distance_k = 182.0 * resolution_scale
            self.args.near_bbox_height = max(
                1, int(round(130 * resolution_scale))
            )
            self.args.far_bbox_height = max(
                1, int(round(85 * resolution_scale))
            )

            detection = detector.detect_yolo_traffic_light(
                self.model,
                detection_frame,
                self.args,
                depth_frame=None,
            )

            if detection is None:
                self.missed_detection_frames += 1
                if (
                    self.last_detection is not None
                    and self.missed_detection_frames
                    <= detector.MAX_MISSED_DETECTION_FRAMES
                ):
                    detection = self.last_detection.copy()
                    detection["class_name"] = "last_valid_detection"
                else:
                    detector.reset_color_filter()
                    detector.reset_distance_filter()
                    self.last_detection = None
            else:
                detection = detector.stabilize_detection_color(detection)
                self.last_detection = detection.copy()
                self.missed_detection_frames = 0

            self.previous_state = detector.print_yolo_detection(
                detection,
                self.previous_state,
            )
            self.publish_detection(detection)

            if self.debug_image_publisher is not None or self.show_window:
                debug_frame = detector.draw_yolo_result(frame, detection)

                if self.debug_image_publisher is not None:
                    debug_message = self.bridge.cv2_to_imgmsg(
                        debug_frame, encoding="bgr8"
                    )
                    debug_message.header = message.header
                    self.debug_image_publisher.publish(debug_message)

                if self.show_window:
                    cv2.imshow("Traffic Light ROS 2", debug_frame)
                    cv2.waitKey(1)
        except CvBridgeError as error:
            self.get_logger().error(f"Could not convert ROS image: {error}")
        except Exception as error:
            self.get_logger().error(f"Traffic-light detection failed: {error}")

    def destroy_node(self):
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
