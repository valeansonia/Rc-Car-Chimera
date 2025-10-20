#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('lane2')

        self.cap = cv2.VideoCapture('/home/arrk-agx/Chimera_master/Chimera/src/lateral_control/lateral_control/reverse.mp4')  # Replace with video path or 0 for webcam
        self.bridge = CvBridge()

        self.publisher_ = self.create_publisher(Image, 'timo', 100)
        self.timer = self.create_timer(0.1, self.timer_callback)  # Publishing at 10Hz

        # Define the new resolution (width, height)
        self.new_resolution = (640, 360)  # Example resolution (640x360)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            # If video ends, reset to the first frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        if ret:
            # Resize the frame to the new resolution
            resized_frame = cv2.resize(frame, self.new_resolution)
            
            # Convert OpenCV image (BGR) to ROS2 Image message
            img_msg = self.bridge.cv2_to_imgmsg(resized_frame, encoding="bgr8")
            self.publisher_.publish(img_msg)
            self.get_logger().info('Publishing resized video frame')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
