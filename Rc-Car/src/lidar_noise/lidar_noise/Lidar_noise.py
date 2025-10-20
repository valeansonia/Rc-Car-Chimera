import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
import random

class LidarNoise(Node):

    def __init__(self):
        super().__init__('lidar_noise')
        self.publisher_ = self.create_publisher(LaserScan, 'base_scan_noise', 10)
        self.subscription = self.create_subscription(LaserScan, 'base_scan', 
        self.lidar_noise_callback, 10)
        self.subscription
        self.std_dev = 0.5
        self.noise = np.random.random_sample(1081,)
        print(self.noise[2] )
        

    def lidar_noise_callback(self, msg):
        msg_noise = LaserScan()
        msg_noise = msg
        iterations = 0
        for x in msg.ranges:
            msg_noise.ranges[iterations] = x + self.noise[iterations] #random.random()
            iterations = iterations + 1  
        msg_noise.ranges = msg.ranges 
        
        self.publisher_.publish(msg_noise)
        
def main(args=None):
    rclpy.init(args=args)

    lidar_noise = LidarNoise()

    rclpy.spin(lidar_noise)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lidar_noise.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()