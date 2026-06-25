# object_detection/detection.py
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection')

        # Create a subscriber to receive messages on the 'detections' topic
        self.subscription = self.create_subscription(
            Detection2DArray,
            'detectnet/detections',
            self.callback,
            10  # QoS profile, adjust as needed
        )
        self.subscription  # prevent unused variable warning

        # Create a publisher to send messages on the 'stop' topic
        self.publisher = self.create_publisher(
            String,
            'stop',
            10  # QoS profile, adjust as needed
        )

    def callback(self, msg):
        vari = Detection2DArray
        vari = msg.detections

        print(type(vari))
        # Check if any tennis ball with confidence more than 80% is detected
        print(msg.detections[0:1])
        if any(msg.detections.score < 0.8 ):
            self.get_logger().info('Stop! Tennis ball detected with high confidence.')

            # Publish a message to the 'stop' topic
            stop_msg = String()
            stop_msg.data = 'Stop'
            self.publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)

    detection_node = DetectionNode()

    rclpy.spin(detection_node)

    detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
