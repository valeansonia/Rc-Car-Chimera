import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32MultiArray, String

from .xbox_controls import DEFAULT_SPEED, DEFAULT_SPEED_INCREMENT


class TrafficLightDrivePolicy(Node):
    def __init__(self):
        super().__init__("traffic_light_drive_policy")

        self.declare_parameter("stop_speed", float(DEFAULT_SPEED))
        self.declare_parameter(
            "slow_speed",
            float(DEFAULT_SPEED + (DEFAULT_SPEED_INCREMENT // 2)),
        )
        self.declare_parameter(
            "go_speed",
            float(DEFAULT_SPEED + DEFAULT_SPEED_INCREMENT),
        )
        self.declare_parameter("red_stop_bbox_height", 150)
        self.declare_parameter("red_slow_bbox_height", 60)

        self.stop_speed = float(self.get_parameter("stop_speed").value)
        self.slow_speed = float(self.get_parameter("slow_speed").value)
        self.go_speed = float(self.get_parameter("go_speed").value)
        self.red_stop_bbox_height = int(
            self.get_parameter("red_stop_bbox_height").value
        )
        self.red_slow_bbox_height = int(
            self.get_parameter("red_slow_bbox_height").value
        )

        self.last_action = None
        self.current_state = "back"
        self.current_bbox = [0, 0, 0, 0]

        self.create_subscription(
            String,
            "/traffic_light/state",
            self.traffic_light_callback,
            10,
        )
        self.create_subscription(
            Int32MultiArray,
            "/traffic_light/bbox",
            self.bbox_callback,
            10,
        )

        self.action_publisher = self.create_publisher(
            String,
            "/traffic_light/drive_action",
            10,
        )
        self.speed_publisher = self.create_publisher(
            Float32,
            "/traffic_light/speed_pwm",
            10,
        )

        self.get_logger().info(
            "Traffic light drive policy started. "
            "green=GO, yellow=SLOW, red=SLOW far and STOP close."
        )

    def bbox_height(self):
        if len(self.current_bbox) != 4:
            return 0

        _, y1, _, y2 = self.current_bbox
        return max(0, y2 - y1)

    def command_for_state(self, state):
        height = self.bbox_height()

        if state == "red":
            if height >= self.red_stop_bbox_height:
                return "RED_STOP_CLOSE", self.stop_speed
            if height >= self.red_slow_bbox_height:
                return "RED_SLOW_FAR", self.slow_speed
            return "RED_SEEN_TOO_FAR", self.go_speed

        if state == "yellow":
            return "YELLOW_SLOW", self.slow_speed

        if state == "green":
            return "GREEN_GO", self.go_speed

        return "NO_TRAFFIC_LIGHT", self.go_speed

    def publish_command(self):
        action, speed = self.command_for_state(self.current_state)

        self.action_publisher.publish(String(data=action))
        self.speed_publisher.publish(Float32(data=speed))

        if action != self.last_action:
            print(
                "LOGGING: Traffic light command works -> "
                f"state={self.current_state}, action={action}, "
                f"speed_pwm={speed:.0f}, bbox_height={self.bbox_height()}"
            )
            self.last_action = action

    def traffic_light_callback(self, msg):
        self.current_state = msg.data.strip().lower()
        self.publish_command()

    def bbox_callback(self, msg):
        self.current_bbox = list(msg.data)
        self.publish_command()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDrivePolicy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
