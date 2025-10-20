#!/usr/bin/env python
import os
import csv
import numpy as np
import rclpy
from rclpy.node import Node
from scipy import signal as signal
from scipy import linalg as linalg
import time
from time import localtime, strftime
from std_msgs.msg import Float32
from std_msgs.msg import Int8
from std_msgs.msg import Int8MultiArray
from ackermann_msgs.msg import AckermannDriveStamped

#Class for the discrete LTI system of the car's lateral dynamic
#needed for calculation of the model-based parameters for the observer matrix L and control matrix K
class LTISystem():
    def __init__(self, logger,  T_s):
        logger.info("Initialize discrete LTI-System of car's lateral dynamic...")
        # continouos LTI-Sys Matrices as x_p = A*x + B*u; y = C * x 
        self.A, self.B, self.C, self.D = self.calc_cont_sys_mat()
        
        #discrete LTI-Sys Matrices with zero-order-hold Sampling
        self.Az, self.Bz, self.Cz, self.Dz = self.calc_disc_sys_mat(self.A, self.B, self.C, self.D, T_s)
        logger.info("System-Initialization done!")
    
    #return the  discrete system matrices
    def get_discrete_mat(self):
        return self.Az, self.Bz, self.Cz, self.Dz
    
    #return the continuous system matrices
    def get_cont_mat(self):    
        return self.A, self.B, self.C, self.D    

    #calc the continuous LTI-System of the Bicycle Model including lateral displacement q and angle theta (see Master Thesis)
    def calc_cont_sys_mat(self):
        v_const = 1.0
        m_car = 5.2
        Jz_car = 0.08
        c_h = 9.752
        c_v = 0.65*c_h
        l_v = 0.196
        l_h = 0.144
        l_car = l_v + l_h
        dist_foresight = 0.6

        A11 = (-c_v-c_h) / (m_car * v_const)
        A12 = (c_h * l_h - c_v * l_v) / (m_car * v_const**2) - 1
        A13 = 0.
        A14 = 0.

        A21 = (c_h * l_h - c_v * l_v) / Jz_car
        A22 = - (c_v * l_v**2 + c_h * l_h**2) / (Jz_car * v_const)
        A23 = 0.
        A24 = 0.
        
        A31 = 0.
        A32 = 1.
        A33 = 0.
        A34 = 0.

        A41 = v_const
        A42 = dist_foresight
        A43 = v_const
        A44 = 0.
    
        A = np.array([[A11, A12, A13, A14], [A21, A22, A23, A24], [A31, A32, A33, A34], [A41, A42, A43, A44]])
        
        B1 = c_v / (m_car * v_const)
        B2 = (c_v * l_v) / (Jz_car)
        B3 = 0.
        B4 = 0.
        
        B = np.array([[B1], [B2], [B3], [B4]])

        C = np.array([[0., 0., 0., 1.]])
        D = np.array([[0]])
        return A, B, C, D

    #calculate the discrete LTI system using zero order hold sampling with T_s as Sampling Time
    def calc_disc_sys_mat(self, A, B, C, D, dt):
        discrete_sys = signal.cont2discrete((A, B, C, D), dt, method='zoh')
        Az, Bz, Cz, Dz, _ = discrete_sys
        
        return Az, Bz, Cz, Dz


#discrete, reduced luenberger observer to estimate the states beta (slip angle), psi_p (yaw rate) and course angle error (theta)
class LuenbergerObserver():
    def __init__(self, lti_sys, T_s, logger):
        self._logger = logger
        self._logger.info("Initialize Luenberger Observer...")

        #get continuous poles for pole placement
        pole_1 = -5
        pole_2 = -6
        pole_3 = -7
        gain = 1
        continuous_obs_poles = gain * np.array([pole_1, pole_2, pole_3])
        discrete_obs_poles = np.exp(continuous_obs_poles * T_s)
        
        #get continuous matrices for implementing the reduced observer model
        A, B, C, D = lti_sys.get_cont_mat()
        self.A11 = np.array([A[-1,-1]])
        self.A21 = A[0:3, -1]
        self.A12 = np.array([A[-1, 0:3]])
        self.A22 = A[0:3, 0:3]		
        self._logger.info("Shape A11: %s" % str(self.A11.shape))
        self.B1 = np.array([B[-1]])
        self.B2 = B[0:-1]
        
        #get discrete matrices to calculate the reduced observer matrix L for discrete time systems
        Az, Bz, Cz, Dz = lti_sys.get_discrete_mat()
        self.Az11 = np.array([Az[-1,-1]])
        self.Az21 = Az[0:3, -1]
        self.Az12 = Az[-1, 0:3]
        self.Az22 = Az[0:3, 0:3]
        
        self.Bz1 = np.array([Bz[-1]])
        self.Bz2 = Bz[0:-1]

        #calculate observer matrix L with pole placement
        self.L_red = self.calc_observer_matrix((self.Az11, self.Az12, self.Az21, self.Az22, self.Bz1, self.Bz2), discrete_obs_poles)
        self.Az12 = np.array([Az[-1, 0:3]])
        self._logger.info("Shape L: %s" % str(self.L_red.shape))
        self._logger.info("Observer initialized!")

    
    #calculate discrete observer matrix for reduced system (Az11, Az12, Az21, Az22, Bz1, Bz2) with pole placement method, so (A22-L*A12) has the desired eigen values
    def calc_observer_matrix(self, red_discrete_sys, discrete_obs_poles):
        Az22 = red_discrete_sys[3] #Az22
        Az12 = np.array([red_discrete_sys[1]]) #Az12
        L_red = signal.place_poles(np.transpose(Az22), np.transpose(Az12), discrete_obs_poles, method='KNV0')
        return np.transpose(L_red.gain_matrix)
        
    def get_obs_matrix(self):
        return self.L_red

    # As the L-vector is calculated for the discrete system, the integrator is just 1/z --> y(k) = u(k-1)
    def calc_integration(self, curr_input, old_input, old_output):
        new_output = old_input #old_output + self.T_s/2 * (curr_input + old_input)
        return new_output
    
    # calculate the states of the current step k
    # old_int_in = input of the discrete integrator of step k-1 
    # old_int_out = output of the discrete integrator of step k-1 
    # old_states = calculated states of step k-1
    # reduced luenberger observer formula (see Thesis):
    #	v_p = (A22 - L*A12)v + (A22 - L*A12)*L*lat_disp + (A21 - L*A11)*lat_disp + (B2 - L*B1)*steering_angle
    # 	states = v + L*y
    def calc_states(self, steering_angle, lateral_disp, old_int_out, old_int_in, old_states):
        sum_1 = np.transpose(np.array([float(lateral_disp) * (self.Az21 - np.matmul(self.L_red, self.Az11))]))
        self._logger.info("Sum1 Dimension: %s" % str(sum_1.shape))
        sum_2 = float(steering_angle) * (self.Bz2 - np.matmul(self.L_red, self.Bz1))
        self._logger.info("Sum2 Dimension: %s" % str(sum_2.shape))
        sum_3 = np.matmul((self.Az22 - np.matmul(self.L_red,self.Az12)), old_states)
        self._logger.info("Sum3 Dimension: %s"% str(sum_3.shape))
        input_integrator = sum_1 + sum_2 + sum_3
        self._logger.info("Input Integrator Dimension: %s" % str(input_integrator.shape))
        output_integrator = self.calc_integration(input_integrator, old_int_in, old_int_out)

        new_states = output_integrator + lateral_disp * self.L_red

        return new_states, input_integrator, output_integrator


#discrete state-feedback control using a reduced luenberger observer       
class StateController(Node):

    def __init__(self):
        super().__init__('steering_control')
        self.get_logger().info("Initialize State-Feedback Controller...")
       
        
        #Parameters##
        self.declare_parameters(
            namespace='',
            parameters=[
                ('v_const', rclpy.Parameter.Type.DOUBLE),
                ('m_car', rclpy.Parameter.Type.DOUBLE),
                ('Jz_car', rclpy.Parameter.Type.DOUBLE),
                ('c_h', rclpy.Parameter.Type.DOUBLE),
                ('c_v', rclpy.Parameter.Type.DOUBLE),
                ('l_v', rclpy.Parameter.Type.DOUBLE),
                ('l_h', rclpy.Parameter.Type.DOUBLE),
                ('l_car', rclpy.Parameter.Type.DOUBLE),
                ('dist', rclpy.Parameter.Type.DOUBLE),
                ('T_s', rclpy.Parameter.Type.DOUBLE),
                ('max_steering_angle', rclpy.Parameter.Type.DOUBLE),
                ('min_steering_angle', rclpy.Parameter.Type.DOUBLE),
                ('anti_windup_constant', rclpy.Parameter.Type.INTEGER),
                ('q1', rclpy.Parameter.Type.DOUBLE),
                ('q2', rclpy.Parameter.Type.DOUBLE),
                ('q3', rclpy.Parameter.Type.DOUBLE),
                ('q4', rclpy.Parameter.Type.INTEGER),
                ('q5', rclpy.Parameter.Type.INTEGER),
                ('weight', rclpy.Parameter.Type.INTEGER),
                ('pole_1', rclpy.Parameter.Type.INTEGER),
                ('pole_2', rclpy.Parameter.Type.INTEGER),
                ('pole_3', rclpy.Parameter.Type.INTEGER),
                ('gain_pole', rclpy.Parameter.Type.INTEGER)
                ])
        
        self.T_s = self.get_parameter("T_s").value
        
        print(type(self.get_parameter("max_steering_angle").value))
        self.u_max = self.get_parameter("max_steering_angle").value
        self.u_min = self.get_parameter("min_steering_angle").value
           
        self.aw_constant = self.get_parameter("anti_windup_constant").value
        self.aw_term = 0
        q1 = self.get_parameter("q1").value
        q2 = self.get_parameter("q2").value
        q3 = self.get_parameter("q3").value
        q4 = self.get_parameter("q4").value
        q5 = self.get_parameter("q5").value # state of I-Control
        weight = self.get_parameter("weight").value
        Q = weight * np.array([[q1, 0, 0, 0, 0], [0, q2, 0, 0, 0], [0, 0, q3, 0, 0], [0, 0, 0, q4, 0], [0, 0, 0, 0, q5]])
        R = 1/(self.u_max**2)
        self.get_logger().info("Q = %s" % str(Q))
        self.get_logger().info("R = %s" % str(R))

        # initialize LTI-System and Observer
        lti_sys = LTISystem( self.get_logger(), self.T_s)
        self.luen_obs = LuenbergerObserver(lti_sys, self.T_s, self.get_logger())

        # control variables
        self.actual_speed = 0.0
        self.actual_steering_angle = 0.0
        self.callback_joy_timestamp = 0
        self.lsa_active = True #False
        self.measurement = False
        self.v_const =  self.get_parameter("v_const").value

        # measurement variable lists
        self.q_measured = []
        self.state_beta = []
        self.state_psi_p = []
        self.state_theta = []
        self.steering_angle_logged = []
        self.timestamp = []
        
        # step k-1 Variables for Observer
        self.old_int_out = np.array([[0], [0], [0]])
        self.old_int_in = np.array([[0], [0], [0]])
        self.old_states = np.array([[0], [0], [0]])
        self.steering_angle = 0.0

        # step k-1 variables for controller
        self.old_lateral_disp = 0
        self.old_integral_state = 0

        # Controller with I-Part ---> the continuous sys matrices expand to a system with order = 5 for calculation of the K-Matrix
        A, B, C, D = lti_sys.get_cont_mat()
        A_I = np.hstack((np.vstack((A, -C)), np.zeros((5,1))))
        B_I = np.vstack((B, np.zeros(1)))
        C_I = np.hstack((C, np.zeros((1,1))))
        
        # calculate discrete system with Integrator
        Az_I, Bz_I, Cz_I, Dz_I = lti_sys.calc_disc_sys_mat(A_I, B_I, C_I, D, self.T_s)
        
        #calculate K Matrix for State-Feedback-Control		
        self.K = self.calc_GainMatrix(Az_I, Bz_I, Q, R)


        #Initialize Subscriber and Publisher
        self.create_subscription(Int8, "/dev/null", self.callback_listen_null, 1)
        self.create_subscription(
            Float32, "lateral_displacement", self.callback_lateral_disp,1)
        
        self.create_subscription(Int8MultiArray, "/lane_info", self.callback_listen_lane_info, 1)
        
        self.control_publisher = self.create_publisher(
            AckermannDriveStamped, "drive", 1) 	
        
        #Ackermann Messages initialization
        self.ack_msg = AckermannDriveStamped()

        self.get_logger().info("====== Regulator/Observer Parameters ======")
        self.get_logger().info("Observer Parameters: %s" % str(self.luen_obs.get_obs_matrix()))	
        self.get_logger().info("Controller Parameters: %s" % str(self.get_control_matrix()))
        self.get_logger().info("State-Feedback Controller initialized!")


        # Save
        self.csv_file_time  = open('steering_time.csv', 'w')
        self.csv_writer_time    = csv.writer(self.csv_file_time)
        self.csv_writer_time.writerow(['time1, time2'])
    
    def __del__(self):
        self.csv_file_drive.close()
        self.csv_file_time.close()
    # Calculation of Controlle-Matrix through solving of the Riccati-Equation --> LQ-Control
    ####### input param ########
    # discrete System Matrices of A and B and Weight Matrices Q,R for the Riccati Equation
    ####### output param #######
    # State-feedback Gain Matrix K, with constant for integrator K[-1]
    def calc_GainMatrix(self, Az, Bz, Q, R):
        S = linalg.solve_discrete_are(Az, Bz, Q, R) # Solution of the Riccati-Equation
        fac1 = np.linalg.inv((R + np.matmul(np.matmul(np.transpose(Bz), S), Bz)))
        K = np.matmul(np.matmul(np.matmul(fac1, np.transpose(Bz)), S), Az) # Regulator Matrix for the State-feedback control
        return K
    
    
        # return the State-Feedback Gain Matrix
        # k0 for beta, k1 for psi_p, k2 for theta, k3 for q, k4 for integral
    def get_control_matrix(self):
        return self.K

        # calculate the states of current step and return them
        ####### input param #######
        # steering_angle: steering angle of last step
        # lateral displacement: current lateral displacement
        # old_int_in = input of the discrete integrator of step k-1 
        # old_int_out = output of the discrete integrator of step k-1 
        # old_states = calculated states of step k-1
        ####### output param #######
        # states: current states value for beta, psi_p and theta
        # input integrator: input of integrator needed for next step
        # output integrator: output of integrator needed for next step
    def get_obs_states(self, steering_angle, lateral_disp, old_int_out, old_int_in, old_states):
        return self.luen_obs.calc_states(steering_angle, lateral_disp, old_int_out, old_int_in, old_states)

        # Calculate the steering angle for current error (lateral displacement)
        ####### input param #######
        # current lateral displacement: curr_lateral_disp [m]
        # lateral displacement of timestep k-1: old_lateral_disp [m]
        # current states calculated of observer beta, psi_p, theta: states
        # integral state x_i of timestep k-1 for Tustin Approximation: old_integral_state
        ####### output param #######
        # steering angle after saturation check: steering_angle_sat [rad]
        # calculated current integral output x_i, needed for next timestep: integral_state
    def calculate_steering_angle(self, curr_lateral_disp, old_lateral_disp, states, old_integral_state):
        self.get_logger().info("Lat DISP Dimension: %s" % str(np.array([[curr_lateral_disp]]).shape))		
        x = np.vstack((states, np.array([[curr_lateral_disp]])))
        self.get_logger().info("States: %s" % str(x.shape))
        K = np.array([self.get_control_matrix()[0, 0:-1]])
        self.get_logger().info("Control Matrix: %s" % str(K.shape))
        proportional = - np.matmul(K, x) # K is (1,4) --> proportional = -K*x (see Thesis)
        integral_state = (old_integral_state + self.T_s/2 * ((-curr_lateral_disp + self.get_aw_term()) + (- old_lateral_disp))) 
        steering_angle = integral_state * (- self.get_control_matrix()[0,-1]) + proportional
        # check if the steering angle > max_steering_angle
        self.get_logger().info("unsaturated steering: %s" % str(steering_angle))
        steering_angle_sat = self.anti_windup(steering_angle.reshape(-1))# integral_state.reshape(-1))
        
        return steering_angle_sat, integral_state
 
    # function to prevent wind up effect due to steering saturation --> back-calculation algorithm
    ####### input param #######
    # steering angle calculated from regulator: steering_angle [rad]
    # integral part of regulator
    ####### output param #######
    # steering_angle after saturation check: steering_angle [rad]
    def anti_windup(self, steering_angle):
        if steering_angle > self.u_max:
            aw_term = (self.u_max - steering_angle) * 1/self.aw_constant
            steering_angle = self.u_max
        elif steering_angle < self.u_min:
            aw_term = (self.u_min - steering_angle) * 1/self.aw_constant
            steering_angle = self.u_min
        else:
            aw_term = 0
        self.set_aw_term(aw_term)

        return steering_angle
        
    # sets the anti windup term which is added to the integrator input to avoid windup effect	
    def set_aw_term(self, aw_term):
        self.aw_term = aw_term

    def get_aw_term(self):
        return self.aw_term

    def set_activation_lsa(self, activation):
        self.lsa_active = activation
    
    def get_activation_lsa(self):
        return self.lsa_active

    def get_last_joy_time(self):
        return self.callback_joy_timestamp

    # callback to topic /dev/null which is used when autonomous driving functions are activated
    # Start the steering control
    def callback_listen_null(self, data):
        self.lsa_active = True
        self.callback_joy_timestamp = self.get_clock().now()

    # callback to topic /lane_info which gives information about lane or no lane, left right or straigt curvature)
    def callback_listen_lane_info(self, lane_info):
        self.lane_found = lane_info.data[0]
        self.course = lane_info.data[1]
        
        self.get_logger().info("lane_found: %s" % str(self.lane_found)) 
        self.get_logger().info("course: %s" % str(self.course)) 

    def reset_values(self):
        # step k-1 Variables for Observer
        self.old_int_out = np.array([[0], [0], [0]])
        self.old_int_in = np.array([[0], [0], [0]])
        self.old_states = np.array([[0], [0], [0]])
        self.steering_angle = 0.0

        # step k-1 variables for controller
        self.old_lateral_disp = 0
        self.old_integral_state = 0

        self.actual_speed = 0
        
        self.q_measured = []
        self.state_beta = []
        self.state_psi_p = []
        self.state_theta = []
        self.steering_angle_logged = []
        self.timestamp = []
    
    def activate_data_measurement(self, activation):
        self.measurement = activation
    
    def get_measurement_activation(self):
        return self.measurement
    
    # Callback function for calculating the needed steering angle
    def callback_lateral_disp(self, q):
        time1 = time.time()

        q = q.data
        if self.get_activation_lsa():
            #if the autonomous driving function is not activated more than 5s then deactivate steering control	
            # issue with dev/null, new joy configurator, uppon activation only publishes once
	    
            self.activate_data_measurement(True)
            #calculate new states with old states
            curr_states, input_integrator_obs, output_integrator_obs = self.get_obs_states(self.steering_angle, q, self.old_int_out, self.old_int_in, self.old_states)
            #calculate new steering angle with old values
            steering_angle_sat, integral_state = self.calculate_steering_angle(q, self.old_lateral_disp, curr_states, self.old_integral_state)
            #add anti-windup term to lateral_displacement for next step anti-windup (see Thesis)
            self.old_integral_state = integral_state #-q + self.get_aw_term()
            #assign new values to old value variable for next step
            self.old_int_in = input_integrator_obs
            self.old_int_out = output_integrator_obs
            self.old_states = curr_states
            self.steering_angle = steering_angle_sat
            self.old_lateral_disp = q
            #publish
            self.get_logger().info("Steering Angle: %s" % str(self.steering_angle))   

            if self.lane_found == 1: 
                # deciding if straight->fast or curve->slow    
                if self.course == 0:
                    velocity = self.get_parameter("v_const").value
                else:
                    velocity = (self.get_parameter("v_const").value) - 0.1

                # Create an AckermannDriveStamped messagetwist.twist.linear.x
                drive_msg = AckermannDriveStamped()
                drive_msg.header.stamp = self.get_clock().now().to_msg()
                self.velocity_driving = velocity 
                drive_msg.drive.speed = self.velocity_driving  # arbitrary speed
                drive_msg.drive.steering_angle = float(steering_angle_sat)  # Calculated steerint
            
            elif self.lane_found == 0:
                # emergency stop
                drive_msg = AckermannDriveStamped()
                drive_msg.header.stamp = self.get_clock().now().to_msg()
                self.velocity_driving =  0.0
                drive_msg.drive.speed = self.velocity_driving  # arbitrary speed
                drive_msg.drive.steering_angle = float(0.0)  # Calculated steerint
                          
            # Publish the drive message
            self.control_publisher.publish(drive_msg)

            # append new values to storage variables
            self.timestamp.append(round(time.time()*1000))
            self.q_measured.append(q)
            self.state_beta.append(curr_states[0])
            self.state_psi_p.append(curr_states[1])
            self.state_theta.append(curr_states[2])
            self.steering_angle_logged.append(steering_angle_sat)

                
            # else: 
                #self.set_activation_lsa(False)
        else: 
            self.get_logger().info("Steering Control inactive")
            # if steering control entered once, then write written data to csv							
            if self.get_measurement_activation():
                self.activate_data_measurement(False)
                str_time = strftime("%Y-%m-%d_%H_%M_%S", localtime())
                folder_name = "LSA_" + str_time
                workspace_path = os.path.dirname(os.path.abspath(__file__)) + "/reports"
                folder_path = os.path.join(workspace_path, folder_name)
                os.mkdir(folder_path)

                csv_file = open(os.path.join(folder_path, "report_" + str_time + ".csv"), "w")
                writer = csv.writer(csv_file)
                writer.writerow(['Timestamp', 'Beta', 'Yawrate', 'Theta', 'Displacement', 'Steering', 'Velocity_Controller', 'Velocity_Driving'])

                for i in range(0,len(self.q_measured)):
                    data = [self.timestamp[i], self.state_beta[i], self.state_psi_p[i], self.state_theta[i], self.q_measured[i], self.steering_angle_logged[i],self.v_const, self.velocity_driving]
                    writer.writerow(data)
                csv_file.close()
                self.reset_values()
        self.csv_writer_time.writerow([time1, time.time()])
        self.csv_file_time.flush()

def main(args=None):
    rclpy.init(args=args)
    node = StateController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()