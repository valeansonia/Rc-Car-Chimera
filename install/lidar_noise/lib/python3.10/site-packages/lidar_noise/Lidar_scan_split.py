#! /usr/bin/env python3
"""
Program to split LaserScan into three parts.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarScanSplit(Node):
    """
    Class for splitting LaserScan into three parts.
    """

    def __init__(self):
        super().__init__('Lidar_Scan_Split')
        self.publisher_1 = self.create_publisher(LaserScan, 'Laser_M90', 10)
        self.publisher_2 = self.create_publisher(LaserScan, 'Laser_M45', 10)
        self.publisher_3 = self.create_publisher(LaserScan, 'Laser_0', 10)
        self.publisher_4 = self.create_publisher(LaserScan, 'Laser_45', 10)
        self.publisher_5 = self.create_publisher(LaserScan, 'Laser_90', 10)
        
        self.update_rate = 50
        self.freq = 1./self.update_rate

        # Initialize variables
        self.scan_data = []

        # Subscribers
        self.subscription = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.subscription
        
        # Timers
        #rospy.Timer(rospy.Duration(self.freq), self.laserscan_split_update)

    '''def lidar_callback(self, msg):
        """
        Callback function for the Scan topic
        """
        self.scan_data = msg '''

    def lidar_callback(self, msg):
        """
        Function to update the split scan topics
        """

        self.scan_data = msg
        scan1 = LaserScan()
        scan2 = LaserScan()
        scan3 = LaserScan()
        scan4 = LaserScan()
        scan5 = LaserScan()

        #scan1 = self.scan_data
        #scan2 = self.scan_data
        #scan3 = self.scan_data
        #scan4 = self.scan_data
        #scan5 = self.scan_data
        
        
        scan1.header = self.scan_data.header
        scan2.header = self.scan_data.header
        scan3.header = self.scan_data.header
        scan4.header = self.scan_data.header
        scan5.header = self.scan_data.header

        scan1.angle_min = self.scan_data.angle_min
        scan2.angle_min = self.scan_data.angle_min
        scan3.angle_min = self.scan_data.angle_min
        scan4.angle_min = self.scan_data.angle_min
        scan5.angle_min = self.scan_data.angle_min

        scan1.angle_max = self.scan_data.angle_max
        scan2.angle_max = self.scan_data.angle_max
        scan3.angle_max = self.scan_data.angle_max
        scan4.angle_max = self.scan_data.angle_max
        scan5.angle_max = self.scan_data.angle_max

        scan1.angle_increment = self.scan_data.angle_increment
        scan2.angle_increment = self.scan_data.angle_increment
        scan3.angle_increment = self.scan_data.angle_increment
        scan4.angle_increment = self.scan_data.angle_increment
        scan5.angle_increment = self.scan_data.angle_increment

        scan1.time_increment = self.scan_data.time_increment
        scan2.time_increment = self.scan_data.time_increment
        scan3.time_increment = self.scan_data.time_increment
        scan4.time_increment = self.scan_data.time_increment
        scan5.time_increment = self.scan_data.time_increment

        scan1.scan_time = self.scan_data.scan_time
        scan2.scan_time = self.scan_data.scan_time
        scan3.scan_time = self.scan_data.scan_time
        scan4.scan_time = self.scan_data.scan_time
        scan5.scan_time = self.scan_data.scan_time

        scan1.range_min = self.scan_data.range_min
        scan2.range_min = self.scan_data.range_min
        scan3.range_min = self.scan_data.range_min
        scan4.range_min = self.scan_data.range_min
        scan5.range_min = self.scan_data.range_min

        scan1.range_max = self.scan_data.range_max
        scan2.range_max = self.scan_data.range_max
        scan3.range_max = self.scan_data.range_max
        scan4.range_max = self.scan_data.range_max
        scan5.range_max = self.scan_data.range_max
	    

        # LiDAR Range
        n = len(self.scan_data.ranges)
        
        scan1.ranges = [float(0)] * n
        scan2.ranges = [float(0)] * n
        scan2.ranges = [float(0)] * n
        scan3.ranges = [float(0)] * n
        scan4.ranges = [float(0)] * n
        scan5.ranges = [float(0)] * n


        # Splitting Block [three equal parts]
        scan1.ranges[100 : 150] = self.scan_data.ranges[100 : 150]
        scan2.ranges[230 : 280] =self.scan_data.ranges[230 : 280]
        scan3.ranges[350 : 415] = self.scan_data.ranges[350 : 415]
        scan4.ranges[485 : 540] = self.scan_data.ranges[485 : 540]
        scan5.ranges[610 : 670] = self.scan_data.ranges[610 : 670]

        # Publish the LaserScan
        self.publisher_1.publish(scan1)
        self.publisher_2.publish(scan2)
        self.publisher_3.publish(scan3)
        self.publisher_4.publish(scan4)
        self.publisher_5.publish(scan5)

    

def main(args=None):
    rclpy.init(args=args)

    lidar_scan_split = LidarScanSplit()

    rclpy.spin(lidar_scan_split)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lidar_scan_split.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
