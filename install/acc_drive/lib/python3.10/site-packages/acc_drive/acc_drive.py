import rclpy
from rclpy.node import Node
import csv
import time

# Messages
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class AccDrive(Node):

    TARGET_SPEED    = 2.0               # set velocity for ACC (m/s) 
    U               = 0.8               # friction coeffitient (adimensional)        
    TR              = 0.8               # reaction time (s)
    G               = 9.8016            # gravitational acceleration (m/s2)

    def __init__(self):
        super().__init__('acc_drive')

        self.dcri           = None      # Initial critical braking distance      
        self.distance       = None      # Initial measured distance 
        self.odom_speed     = 0.0       # Initial Speed, since run is before callback_vel it does not find it. 

        # Subcribers
        self.sub_vel = self.create_subscription(Odometry, '/odom', self.callback_vel, 1)

        self.sub_depth_info = self.create_subscription(
            Float32,
            'depth_info',
            self.callback_depth,
            1)  # Depth subscriber queue sizepublish

        # Publishers
        self.publisher = self.create_publisher(
            AckermannDriveStamped,
            'drive',
            1)  # Drive publisher queue size

        # Save
        self.csv_file_drive = open('drive_data.csv', 'w')
        self.csv_file_time  = open('drive_time.csv', 'w')

        self.csv_writer_drive   = csv.writer(self.csv_file_drive)
        self.csv_writer_time    = csv.writer(self.csv_file_time)

        self.csv_writer_drive.writerow(['Distance', 'Dcri', 'Odom Speed', 'Command Speed'])
        self.csv_writer_time.writerow(['time, dt'])


    def __del__(self):
        self.csv_file_drive.close()
        self.csv_file_time.close()


    ####################
    #   CALCULATIONS   # 
    ####################

    def calculate_dcri(self, speed):
        return speed * (self.TR + (speed / (2*self.U*self.G)))

    def run (self):        
        
        # Recalculate command speed based on critical braking distance formula
        a = 1.0
        b = 2*self.U*self.G*self.TR
        c = -2*self.U*self.G*self.distance
        
        speed_est = (-(b)+((b)**2-4*(a*c))**0.5)/(2*a)
        
        # Update Values
        self.command_speed    = min(speed_est, self.TARGET_SPEED)  # Apply speed limit
        self.dcri           = self.calculate_dcri(self.odom_speed) #Critical distance is calculated with odom_speed

        return


    #################
    #   CALLBACKS   # 
    #################

    def callback_depth (self, msg):        
        
        time1 = time.time()
        # Assuming the depth info represents the distance to an object in front of the car
        self.distance = msg.data
        
        self.run()
        self.publish()
        
        self.csv_writer_time.writerow([time1, time.time() - time1])
        self.csv_file_time.flush()
        
        return

    def callback_vel (self, msg):
        self.odom_speed = msg.twist.twist.linear.x
        return


    ###############
    #   PUBLISH   # 
    ###############

    def publish (self):

        # Create an AckermannDriveStamped messagetwist.twist.linear.x
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = self.command_speed  # publish the calculated speed based on dcri
        drive_msg.drive.steering_angle = 0.0  # No steering angle
        
        # Publish the drive message
        self.publisher.publish(drive_msg)

        self.csv_writer_drive.writerow([self.distance, self.dcri, self.odom_speed, self.command_speed])
        self.csv_file_drive.flush()

        self.get_logger().info(
            f"Distance: {self.distance:.2f}, DCRI: {self.dcri:.2f}, Speed: {self.odom_speed:.2f}, Final Speed: {self.command_speed:.2f}"
        )

        return


def main(args=None):
    rclpy.init(args=args)
    node = AccDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
