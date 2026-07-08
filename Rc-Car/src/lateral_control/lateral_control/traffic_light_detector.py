from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras


MODEL_FILE_NAMES = [
    "20260701-08091782893391-all-images-mobilenetv2-Adam.h5",
]


class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__("traffic_light_detector")

        self.declare_parameter("image_topic", "/ZEDcam/image_raw")
        self.declare_parameter("model_path", "")
        self.declare_parameter("labels", "back,green,red,yellow")
        self.declare_parameter("stop_labels", "red")
        self.declare_parameter("confidence_threshold", 0.75)
        self.declare_parameter("input_width", 224)
        self.declare_parameter("input_height", 224)
        self.declare_parameter("stable_frames", 2)

        self.image_topic = self.get_parameter("image_topic").value
        self.labels = self._parse_csv_parameter("labels")
        self.stop_labels = set(self._parse_csv_parameter("stop_labels"))
        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        self.stable_frames = max(1, int(self.get_parameter("stable_frames").value))

        self.bridge = CvBridge()
        self.model = keras.models.load_model(self._resolve_model_path())

        self.candidate_label = "unknown"
        self.candidate_count = 0
        self.current_label = "unknown"
        self.current_confidence = 0.0

        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.state_publisher = self.create_publisher(
            String,
            "/traffic_light/state",
            10,
        )
        self.confidence_publisher = self.create_publisher(
            Float32,
            "/traffic_light/confidence",
            10,
        )
        self.stop_publisher = self.create_publisher(
            Bool,
            "/traffic_light/stop",
            10,
        )
        self.action_publisher = self.create_publisher(
            String,
            "/traffic_light/action",
            10,
        )

        self.get_logger().info(
            "Traffic light detector started on topic "
            f"{self.image_topic} with labels {self.labels}"
        )

    def _parse_csv_parameter(self, parameter_name):
        raw_value = self.get_parameter(parameter_name).value
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def _resolve_model_path(self):
        configured_path = self.get_parameter("model_path").value
        if configured_path:
            model_path = Path(configured_path).expanduser()
            if model_path.exists():
                return str(model_path)
            raise FileNotFoundError(f"Model not found: {model_path}")

        project_root = Path(__file__).resolve().parents[3]
        search_dirs = [
            Path("/home/wsadmin/Downloads"),
            project_root / "src" / "model",
            Path.cwd() / "src" / "model",
            Path(__file__).resolve().parents[2] / "model",
        ]

        candidates = [
            search_dir / model_file_name
            for search_dir in search_dirs
            for model_file_name in MODEL_FILE_NAMES
        ]

        for candidate in candidates:
            if candidate.exists():
                self.get_logger().info(f"Using model: {candidate}")
                return str(candidate)

        raise FileNotFoundError(
            "Traffic light model was not found. Set the ROS parameter "
            "'model_path' to the full path of your .h5 model."
        )

    def preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def predict_label(self, frame):
        input_tensor = self.preprocess(frame)
        predictions = self.model.predict(input_tensor, verbose=0)

        class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_index])

        if class_index >= len(self.labels):
            self.get_logger().warn(
                f"Model returned class index {class_index}, but only "
                f"{len(self.labels)} labels are configured."
            )
            return "unknown", confidence

        if confidence < self.confidence_threshold:
            return "unknown", confidence

        return self.labels[class_index], confidence

    def update_stable_prediction(self, label, confidence):
        if label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = label
            self.candidate_count = 1

        if self.candidate_count >= self.stable_frames:
            self.current_label = label
            self.current_confidence = confidence

    def publish_prediction(self):
        should_stop = self.current_label in self.stop_labels
        action = self.action_for_label(self.current_label)

        self.state_publisher.publish(String(data=self.current_label))
        self.confidence_publisher.publish(Float32(data=self.current_confidence))
        self.stop_publisher.publish(Bool(data=should_stop))
        self.action_publisher.publish(String(data=action))

        self.get_logger().info(
            "traffic_light="
            f"{self.current_label}, confidence={self.current_confidence:.2f}, "
            f"action={action}, stop={should_stop}"
        )

    def action_for_label(self, label):
        if label == "red":
            return "STOP"
        if label == "yellow":
            return "SLOW"
        if label == "green":
            return "GO"
        return "NO_TRAFFIC_LIGHT"

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            label, confidence = self.predict_label(frame)
            self.update_stable_prediction(label, confidence)
            self.publish_prediction()
        except CvBridgeError as error:
            self.get_logger().error(f"Could not convert ROS image: {error}")
        except Exception as error:
            self.get_logger().error(f"Traffic light detection failed: {error}")


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
