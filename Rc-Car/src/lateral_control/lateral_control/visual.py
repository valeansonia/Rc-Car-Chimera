import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()

        """# Create subscribers for the three topics
        self.hd_cam_subscriber = self.create_subscription(
            Image,
            '/calibration_image_HDwebCam',
            self.hd_cam_callback,
            10)"""
        
        self.zed_cam_subscriber = self.create_subscription(
            Image,
            '/dashboard', #/calibration_image_ZEDcam
            self.zed_cam_callback,
            10)

        self.lane_detection_subscriber = self.create_subscription(
            Image,
            '/lane_detection_ZEDcam',
            self.lane_detection_callback,
            10)

    """def hd_cam_callback(self, msg):
        # Convert ROS Image message to OpenCV format and display it
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("HD Webcam", frame)
        cv2.waitKey(1)"""

    def resize(self, image, desired_width):
        #get the original dimensions
        (h, w) = image.shape[:2]

        #desired width
        new_width = desired_width
        
        #Calculate the aspect ratio
        aspect_ratio = h / w 
        new_height = int(new_width * aspect_ratio)

        return cv2.resize(image, (new_width, new_height))


    def zed_cam_callback(self, msg):
        # Convert ROS Image message to OpenCV format and display it
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #print()
        cv2.imshow("ZED Camera", self.resize(frame, 1080))
        cv2.waitKey(1)

    def lane_detection_callback(self, msg):
        # Convert ROS Image message to OpenCV format and display it
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Lane Detection", self.resize(frame, 1080))
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown
        cv2.destroyAllWindows()
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()