import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np

pi = 3.14159265358979323846

class FrontLidarDetector(Node):
    def __init__(self):
        super().__init__('front_lidar_detector')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.threshold_high = 1.5  # meters
        self.threshold_low = 0.25   # meters
        self.min_cluster_size = 3  # beams per object
        self.front_angle_limit = 30 * pi / 180  # ±30° in radians

        self.obstacle_pub = self.create_publisher(Bool, '/obstacle_detected', 10)

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        n = len(ranges)

        # compute angle for each beam
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        # only consider beams in front of the car
        front_mask = np.abs(angles) <= self.front_angle_limit
        front_indices = np.where((ranges < self.threshold_high) & front_mask)[0]

        if len(front_indices) == 0:
            return  # nothing close in front

        # cluster consecutive beams
        clusters = np.split(front_indices, np.where(np.diff(front_indices) > 1)[0]+1)

        for cluster in clusters:
            obstacle_found = False
            if len(cluster) >= self.min_cluster_size:
                cluster_distances = ranges[cluster]
                min_idx = cluster[np.argmin(cluster_distances)]
                distance = ranges[min_idx]
                angle = angles[min_idx]
                
                distance_m = float(distance)

                # if obstacle in front within thresholds
                if(distance_m < self.threshold_high and distance_m > self.threshold_low):
                    obstacle_found = True 
                    print(f"Object detected in front at {distance:.3f} m, angle {angle:.2f} rad ({angle*180/pi:.1f}°)")
                    #print("No object in front")
                else:
                    print(f"No object in front")    
            else:
                print("No object in front")
                
            self.obstacle_pub.publish(Bool(data=obstacle_found))    


def main(args=None):
    rclpy.init(args=args)
    detector = FrontLidarDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
