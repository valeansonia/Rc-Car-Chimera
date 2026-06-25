import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
import time
import numpy as np

# bring in your existing helper funcs/constants
from .xbox_controls import DEFAULT_SPEED, SPEED_STEP, DEFAULT_STEERING, map_steering_value

class VideoPublisher(Node):
    def __init__(self, video_path):
        super().__init__('video_publisher')

        self.publisher_angle = self.create_publisher(Float32, '/steering_speed_pca', 10)
        self.publisher_obstacle = self.create_publisher(Bool, '/obstacle_detected', 10)

        # optional: publish radius
        self.curve_timo_publisher = self.create_publisher(Float32, '/curve_radius', 10)

        # conversion constants for calc_radius
        self.pixel2meter = 0.01   # 1 pixel = 1 cm (example)
        self.x_dim_foi = 10       # arbitrary FoI width

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        self.last_true_time = 0
        self.true_hold_time = 2  # seconds

        # run at ~20 FPS
        self.timer = self.create_timer(0.05, self.timer_callback)


    def calc_radius(self, info_matrix):
        first_center_x = info_matrix[0,0]    
        first_center_y = info_matrix[0,1]    
        last_center_x = info_matrix[1,0]
        last_center_y = info_matrix[1,1]

        h = first_center_y - last_center_y
        off = first_center_x - last_center_x
        gamma = np.arctan(h/off)
        alpha = np.pi - gamma*2
        radius = h/np.sin(alpha)

        if radius > 200000:
            radius = 1000*self.x_dim_foi

        radius_m = radius * self.pixel2meter

        # Publish radius
        radius_msg = Float32()
        radius_msg.data = radius_m
        self.curve_timo_publisher.publish(radius_msg)

        return radius, radius_m

    def detect_lane_and_obstacles(self, frame):
        """
        Detect lanes from video frame and compute steering angle using calc_radius.
        """
        # --- example lane detection (replace with your real logic) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Hough lines (simplified)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)
        if lines is None:
            return 755, False  # 755 = no road detected

        # compute two points for calc_radius
        # pick first and last line centers
        info_matrix = np.zeros((2,2))
        info_matrix[0] = np.mean(lines[0,:,0:2], axis=0)  # first line center
        info_matrix[1] = np.mean(lines[-1,:,0:2], axis=0)  # last line center

        # get radius
        radius, radius_m = self.calc_radius(info_matrix)

        # convert radius into a steering angle (example mapping)
        angle = float(radius_m)


        # for simplicity, no obstacle detection from video yet
        obstacle_detected = False

        return angle, obstacle_detected

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video reached.")
            rclpy.shutdown()
            return

        now = time.time()

        # ---- run your lane/obstacle detection ----
        angle, obstacle_detected = self.detect_lane_and_obstacles(frame)

        # obstacle "hold" logic
        if obstacle_detected:
            self.last_true_time = now
            obstacle_state = True
        else:
            if now - self.last_true_time >= self.true_hold_time:
                obstacle_state = False
            else:
                obstacle_state = True

        # publish results
        self.publisher_angle.publish(Float32(data=angle))
        self.publisher_obstacle.publish(Bool(data=obstacle_state))

        print(f"LOGGING (video): Angle={angle}, Obstacle= NU MA INTERESEAZA")

        # optional: display video
        cv2.imshow("Video Autonomous", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    video_path = "/home/arrk-adas/Desktop/test.mp4"
    node = VideoPublisher(video_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()