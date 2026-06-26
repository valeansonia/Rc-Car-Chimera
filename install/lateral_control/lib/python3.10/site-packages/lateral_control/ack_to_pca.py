import struct
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray
#import Adafruit_PCA9685
from adafruit_pca9685 import PCA9685 
#from board import SCL, SDA, I2C
import board
import busio
from std_msgs.msg import Float32, Bool

import rclpy
import os 

from .xbox_controls import *

# event from the Xbox
xbox_device = "/dev/input/event11"

try:
    fd = os.open(xbox_device, os.O_RDONLY)
except OSError as e:
    print(f"LOGGING: Error opening {xbox_device}: {e}")
    exit(1)

print("LOGGING: Listening for Xbox controller input...")

FORMAT = 'llHHI'
EVENT_SIZE = struct.calcsize(FORMAT)


class AckToPca(Node):
    def __init__(self):
        super().__init__('ack_to_pca')

        
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c, address=0x40)
        self.pca.frequency = 50

        self.last_true_time = 0
        self.true_hold_time = 2

        self.last_direction = "STRAIGHT"
        # control subscribtion
        self.create_subscription(Float32, '/steering_speed_pca', self.pca_callback2, 10)
        self.create_subscription(Int8MultiArray, "/lane_info", self.lane_info_callback, 100)
        
        # obstacle detection subscription
        self.create_subscription(Bool, '/obstacle_detected', self.obstacle_callback, 10)
        self.obstacle_in_front = False

        self.start_time = None
        self.running_fast = False

        self.readings = [] 
        self.window_size = 0.15

        self.vehicle_velocity = DEFAULT_SPEED
        self.lane_detected = False

    # setters
    def set_speed(self, value):
        self.pca.channels[SPEED_PIN].duty_cycle = value
        print(f"LOGGING: Current speed: {value}")

    def set_steering(self, value):
        self.pca.channels[STEERING_PIN].duty_cycle = value
        print(f"LOGGING: Steering set to: {value}")


    def lane_info_callback(self, msg):
        if msg.data[0] == 0: # no lane found TODO: or obstacle
            print("Lane not found... stopping longitudinal...")
            self.lane_detected = False
            self.vehicle_velocity = DEFAULT_SPEED # 5200 -> too low for longitudinal
        else: # set_speed if msg.data[0] == 1 and no obstacle
            print("Lane found... strating longitudinal...")
            self.lane_detected = True
            self.vehicle_velocity = DEFAULT_SPEED + DEFAULT_SPEED_INCREMENT

    # autonomous logic
    def obstacle_callback(self, msg):
        now = time.time()
        if msg.data:
            self.obstacle_in_front = True
            self.last_true_time = now
        else:
            # only reset to false if enough time has passed
            if now - self.last_true_time >= self.true_hold_time:
                self.obstacle_in_front = False

# pca_callback_lidar function not used -> TODO: remove
    def pca_callback_lidar(self, msg):
        if self.obstacle_in_front:
            self.set_speed(DEFAULT_SPEED)
        else:
            self.set_speed(DEFAULT_SPEED + SPEED_STEP)


    def pca_callback2(self, msg):
        """
        Smoothing with short-term averaging of last 0.15s readings.
        So far, it's the best. 
        Logic:
        - Collect readings in last 0.15s
        - If more left readings → average left
        - If more right readings → average right
        - If tie → average all
        - Map chosen value to steering
        - Handle warm-up and obstacle as before
        """
        now = time.time()
        lane_value = msg.data  # -3 to 3

        if not self.running_fast and not self.obstacle_in_front:
            self.running_fast = True
            self.start_time = now

        if self.running_fast and self.start_time and (now - self.start_time < 0.67):
            print("LOGGING: Skipping steering update during warm-up")
            return

        # collect readings 
        self.readings.append((now, lane_value))
        # keep only last 0.15s readings
        self.readings = [(t, v) for (t, v) in self.readings if now - t <= self.window_size]

        chosen_value = lane_value 

        if len(self.readings) >= 3:
            left_values = [v for _, v in self.readings if v < 0]
            right_values = [v for _, v in self.readings if v > 0]

            if len(left_values) > len(right_values) and left_values:
                chosen_value = sum(left_values) / len(left_values)  # avg left
                print(f"LOGGING: Averaged LEFT → {chosen_value:.2f}")
            elif len(right_values) > len(left_values) and right_values:
                chosen_value = sum(right_values) / len(right_values)  # avg right
                print(f"LOGGING: Averaged RIGHT → {chosen_value:.2f}")
            else:
                # if tie, take average of all values:Carinaaa/Rc-Car-Chime
                chosen_value = sum(v for _, v in self.readings) / len(self.readings)
                print(f"LOGGING: Averaged MIXED → {chosen_value:.2f}")

        # map value to steering
        if chosen_value < -3 or chosen_value > 3:
            steer_val = DEFAULT_STEERING
            print("LOGGING: Straight road detected")
        else:
            scale = 1 - abs(chosen_value) / 3
            if chosen_value < 0:
                steer_val = DEFAULT_STEERING + int(scale * 2720)
            else:
                steer_val = DEFAULT_STEERING - int(scale * 2720)

        self.set_steering(steer_val)

        if self.obstacle_in_front or (not self.lane_detected):
            self.vehicle_velocity = DEFAULT_SPEED
            self.running_fast = False
        elif (not self.obstacle_in_front) and self.lane_detected:
            self.vehicle_velocity = DEFAULT_SPEED + DEFAULT_SPEED_INCREMENT
            self.running_fast = True

        self.set_speed(self.vehicle_velocity)


   

    # manual logic
    def handle_controller_input(self, code, value):
        if code == XBOX_GAS_TRIGGER:
            # check release
            if value == 0:
                self.set_speed(DEFAULT_SPEED)
                print("LOGGING: Gas trigger released → Resetting speed.")
            else:
                # set speed based on gas trigger value
                speed_val = DEFAULT_SPEED + value + OFFSET_FROM_STOP_TO_START_MOVING
                self.set_speed(speed_val if speed_val <= DEFAULT_FORWARD_FULL_SPEED_LIMITED else DEFAULT_FORWARD_FULL_SPEED_LIMITED)

                # if value > 1022:
                # else:
                #     self.set_speed(DEFAULT_SPEED)

        elif code == XBOX_BRAKE_TRIGGER:
            # check release
            if value == 0:
                self.set_speed(DEFAULT_SPEED)
                print("LOGGING: Brake trigger released → Resetting speed.")
            else:
                # set speed based on brake trigger value
                speed_val = DEFAULT_SPEED - value # TODO: 
                self.set_speed(speed_val if speed_val >= DEFAULT_REVERSE_FULL_SPEED_LIMITED else DEFAULT_REVERSE_FULL_SPEED_LIMITED)

                # if value > 1022:
                #     self.set_speed(DEFAULT_SPEED - 2 * SPEED_STEP)
                # elif value > 300:
                #     self.set_speed(DEFAULT_SPEED - SPEED_STEP)
                # else:
                #     self.set_speed(DEFAULT_SPEED)

        else:
            if value != 0:        
                if code == XBOX_LEFT_STICK_X:
                    percentage = remap_stick_value_to_0_100_range(value, deadzone=3000) / 100
                    steer_val = DEFAULT_STEERING + int(2720 * percentage - 1360)
                    self.set_steering(steer_val)

                elif code == XBOX_BUTTON_Y:
                    self.set_speed(DEFAULT_SPEED)

                elif code == XBOX_BUTTON_START:
                    self.set_speed(4990)
                    self.set_steering(4990)
                    print("LOGGING: Normal values. Ending program.")
                    return False  # signal to stop

        return True  # continue running
        
def main(args=None):
    rclpy.init(args=args)
    ack_to_pca = AckToPca()

    # set default values 
    ack_to_pca.set_speed(DEFAULT_SPEED)
    ack_to_pca.set_steering(DEFAULT_STEERING)

    # start in manual mode
    manual_mode = True
    print("LOGGING: Starting in Manual Mode. Press Home for exiting or A to toggle into Autonomous Mode.")

    try:
        while rclpy.ok():
            # always check for input from controller
            import select
            rlist, _, _ = select.select([fd], [], [], 0.01)  # non-blocking wait
            if rlist:
                data = os.read(fd, EVENT_SIZE)
                _, _, type_, code, value = struct.unpack(FORMAT, data)

                if type_ in (1, 3):  # 1 = key/button, 3 = axis
                    if code == XBOX_BUTTON_A and type_ == 1 and value == 1:
                        manual_mode = not manual_mode
                        #ack_to_pca.set_speed(DEFAULT_SPEED)
                        print("\nLOGGING: Toggled mode -> ", "Manual" if manual_mode else "Autonomous")
                        #time.sleep(1.5)

                    if manual_mode:
                        print(f"LOGGING: Code: 0x{code:02X} | Value: {value} → ", end='')
                        if not ack_to_pca.handle_controller_input(code, value):
                            break

            # autonomous mode -> run spin_once to process ROS messages
            if not manual_mode:
                rclpy.spin_once(ack_to_pca, timeout_sec=0.01)

    except KeyboardInterrupt:
        print("LOGGING: Keyboard interrupt, shutting down.")

    finally:
        ack_to_pca.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
