#!/usr/bin/env python

import rclpy 
from rclpy.node import Node 
import cv2 as cv 
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import csv 
import time 
from std_msgs.msg import Float32 
from std_msgs.msg import Int8MultiArray
from sensor_msgs.msg import Image 
from ackermann_msgs.msg import AckermannDriveStamped
import time
from time import localtime, strftime

class LaneDetection(Node):

	def __init__(self): 
		super().__init__('lane_detection')
				
		# declare parameters
		self.declare_parameter('develop', False)	
		self.declare_parameter('stopwatch', False)
		self.declare_parameter('thresh_HDwebCam', 180)
		self.declare_parameter('color_white', (255,255,255))
		self.declare_parameter('color_red', (0,0,255))
		self.declare_parameter('color_green', (0,255,0))
		self.declare_parameter('color_blue', (255,0,0))
		self.declare_parameter('color_purple', (155,25,220))

		# retrieve parameters
		self.develop_flag 		= self.get_parameter('develop').value			
		self.stopwatch_flag		= self.get_parameter('stopwatch').value
		self.thresh_HDwebCam	= self.get_parameter('thresh_HDwebCam').value
		self.color_white		= self.get_parameter('color_white').value
		self.color_red			= self.get_parameter('color_red').value
		self.color_green		= self.get_parameter('color_green').value
		self.color_blue			= self.get_parameter('color_blue').value
		self.color_purple		= self.get_parameter('color_purple').value
	
		self.bridge = CvBridge()

		# Subcribers
		self.image_sub = self.create_subscription(Image, '/HDwebCam/image_raw', self.callback_lat_disp, 10)
		 
		# Publishers
		self.control_value = self.create_publisher(Float32, "lateral_displacement", 100)
		
		if self.develop_flag:
			self.image_pub20 = self.create_publisher(Image, "lane_position_HDwebCam", 10)
			self.image_pub21 = self.create_publisher(Image, "scanrow_bgr_HDwebCam", 10)
			self.image_pub23 = self.create_publisher(Image, "scanrow_bw_HDwebCam", 10)
			self.image_pub30 = self.create_publisher(Image, "calibration_image_HDwebCam", 10)

		# load csv file to write timestamps				
		if self.stopwatch_flag:
			self.csv_file_time  = open('lane_detection_time.csv', 'w')
			self.csv_writer_time    = csv.writer(self.csv_file_time)
			self.csv_writer_time.writerow(['time_calc_params','time_callback_lat_disp'])
			
		# initialising Tokens
		self.init_calc_did_run = False
		self.lane_is_detected = True
		print("initialisation done")
	
	def __del__(self):
		self.csv_file_time.close()

	def callback_lat_disp(self,data):
		time1 = time.time()

		if self.develop_flag:
			print("--------------------------------------------------")	
			print("NEW ITERATION In_Lane_Position")
			print("--------------------------------------------------")

		
		#timer
		start_callback_lat_disp = time.time()

		self.t_past = int(round(time.time()*1000))	#time in ms getrs round and converted in sec	
		try:
        	 #camera picture is converted to the cv2 type so that in can be processed by OpenCV functions
			cv_image_orig = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e: 
			print(e) #if there is an error it will be printed out

		
		#timer
		if self.stopwatch_flag:
			start_calc_params = time.time()

		#Define area and positions for lane detection
		if self.init_calc_did_run == False:
			# number of pixels that are summed up to check for lane
			self.sum_hight = 20
			self.lane_thresh = 255*(self.sum_hight-6)

			# get the dimensions of the ZED Image from the cv_image_orig
			self.get_dimenson_of_HDwebCam_image(cv_image_orig)

			#calculating the conversion factor from Pixel to Meter and reverse
			self.calc_pixel2meter()
		
			# calculating the car center including the camera offset
			self.calc_car_center_in_images()

			# calculating the lane width
			self.calc_lane_width()

			# initialize values for moving average
			self.init_movingaverage()

			self.previous_center = self.car_center_img
			
			
			print("y_dim_min:_______________", self.y_dim_min)
			print("y_dim_center:____________", self.y_dim_center)
			print("y_dim_max:_______________", self.y_dim_max)	
			print("x_dim_min:_______________", self.x_dim_min)
			print("x_dim_center:____________", self.x_dim_center)
			print("x_dim_max:_______________", self.x_dim_max)
			print("car_center_img:__________", self.car_center_img)
			
			print("image measures initialised")	
			self.init_calc_did_run = True
			#time.sleep(3)

			# timer
			if self.stopwatch_flag:
				stop_calc_params = time.time()
				self.time_calc_params = stop_calc_params - start_calc_params
			
		# mein function of this node
		self.fast_lane_detct(cv_image_orig)

		# timer
		if self.stopwatch_flag:
			stop_callback_lat_disp = time.time()
			self.time_callback_lat_disp = stop_callback_lat_disp - start_callback_lat_disp
			print("time_callback_lat_disp ->", self.time_callback_lat_disp)
			self.csv_writer_time.writerow([self.time_calc_params, self.time_callback_lat_disp])
			self.csv_file_time.flush()


	


	
	#################################################
	# READING THE DIMENSIONS OF THE ZED INPUT IMAGE #
	#################################################

	def get_dimenson_of_HDwebCam_image(self, cv_image_orig):
		# Dimensions of the original ZED Image
		self.y_dim_min = 0
		self.y_dim_center = cv_image_orig.shape[0] // 2
		self.y_dim_max = cv_image_orig.shape[0] #376p up to 1242p
		self.x_dim_min = 0
		self.x_dim_center = cv_image_orig.shape[1] // 2 #maybe set to a certain Pixel length
		self.x_dim_max = cv_image_orig.shape[1]
		



	#####################################################################
	# CALCULATING THE CONVERSION FACTOR FROM PIXEL TO METER AND REVERSE #
	#####################################################################	
		
	def calc_pixel2meter(self):
		# calculating from pixel to meter 		
		self.pixel2meter = 0.580 / self.x_dim_max # HDWebCam camera:meters per pixel; 580 mm width measured at 16cm distance (from ZED lens) at the first visible pixle row for 1242p/// 0.49/640 #Logitech camera ///
		self.meter2pixel = 1 / self.pixel2meter




	##############################################################
	# CALCULATING THE X-POSITION OF THE CAR CENTER IN THE IMAGES #
	##############################################################

	def calc_car_center_in_images(self):
		
		#self.offset_lense = 0.060 #0.06 #ZED camera horizontal offset from center of front axis, +6cm for left lens
		self.offset_lense = 0.000 #0.06 #ZED camera horizontal offset from center of front axis, +6cm for left lens
		self.car_center_img = np.int32((self.x_dim_max / 2) + (self.offset_lense * self.meter2pixel)) # half of the image plus the offset of the lens (60 mm)


	##############################################
	# CALCULATING THE WIDTH OF THE LANE IN PIXEL #
	##############################################

	def calc_lane_width(self):
		
		self.lane_width = 420 # measured value
		self.lane_half_width = np.int32(self.lane_width / 2)
		



	#################################################
	# INITALIZING THE VALUES FOR THE MOVING AVERAGE #
	#################################################
	def init_movingaverage(self):
		self.lateral_displacement_ppre = 0
		self.lateral_displacement_pre = 0




	###################################
	# LANE DETECTION OF TO CENTER CAR #
	###################################
	
	def fast_lane_detct(self, cv_image_orig):

		#timer
		if self.stopwatch_flag:
			start_fastlanedetct = time.time()
				
		# Frame of interest of the left zed imgage - cropded frame of picture for the lanedetection
		scanrow_img = cv_image_orig[self.y_dim_center - self.sum_hight:self.y_dim_center, self.x_dim_min:self.x_dim_max] 
		
		#timer
		if self.stopwatch_flag:
			start_cv_brg2hls = time.time()

		# gbr to hls convertion - turns pixel value from rgb into 0:hue,1:lightness 2:saturation 
		scanrow_img_hls = cv.cvtColor(scanrow_img, cv.COLOR_BGR2HLS) 
		
		#timer		
		if self.stopwatch_flag:	
			stop_cv_brg2hls = time.time()
			time_cv_brg2hls = stop_cv_brg2hls - start_cv_brg2hls
			print("time_cv_brg2hls ->", time_cv_brg2hls)

		#timer
		start_thresh = time.time()

		# theashold to get a black and white image
		_ , scanrow_img_hls_threshold = cv.threshold(scanrow_img_hls[:,:,1], self.thresh_HDwebCam, 255, cv.THRESH_BINARY) #original(100,255) #dimensions [y_pixel,x_pixel,informations], sets a threshold on the grayvalues to make the picture black and white
		
		#timer
		if self.stopwatch_flag:
			stop_thresh = time.time()
			time_thresh = stop_thresh - start_thresh
			print("time_thresh ->", time_thresh)

		#timer
		if self.stopwatch_flag:
			start_blur = time.time()
		
		# blur image
		blur_aoi = cv.GaussianBlur(scanrow_img_hls_threshold, (3,3), 0) #blurring the black and white picture
		
		#timer		
		if self.stopwatch_flag:	
			stop_blur = time.time()
			time_blur = stop_blur - start_blur
			print("time_blur ->", time_blur)

		if self.stopwatch_flag:
			stop_laneextraction = time.time()
			time_laneextraction = stop_laneextraction - start_fastlanedetct
			print("time_laneextraction", time_laneextraction)
					
		# define default center
		default_center = self.car_center_img

		# scan window for left lane
		window_left_begin = self.x_dim_min
		window_left_end = default_center
	
		# scan window for right lane 
		window_right_begin = default_center
		window_right_end = self.x_dim_max

		# initialising token for lane found status
		left_lane_found = False 	
		right_lane_found = False 	
		in_left_window = True
		in_right_window = True

		# initialising counter for for shifting the scanposition in the scanrow
		shift_left = 0 				
		shift_right = 0 			

		# left lane scan
		while left_lane_found == False and in_left_window == True:
			
			# the pixel index where the lane is searched					
			scan_index_left =  window_left_end - shift_left 

			# searching for white pixels in the window
			if scan_index_left >= window_left_begin and scan_index_left <= window_left_end:
									
				scan_left = blur_aoi[:, scan_index_left - 1:scan_index_left]
				scan_left_sum = np.sum(scan_left, axis=0)
				if scan_left_sum >= self.lane_thresh:
					left_lane_pixel = scan_index_left
					left_lane_found = True
			else:
				in_left_window = False

			shift_left += 1

		# right lane scan		
		while right_lane_found == False and in_right_window == True:
			
			# the pixel index where the lane is searched					
			scan_index_right = window_right_begin + shift_right 
	
			# searching for white pixels in the window
			if scan_index_right >= window_right_begin and scan_index_right < window_right_end:
				
				scan_right = blur_aoi[:, scan_index_right :scan_index_right + 1]
				scan_right_sum = np.sum(scan_right, axis=0)
				if scan_right_sum >= self.lane_thresh:
					right_lane_pixel = scan_index_right
					right_lane_found = True
			else:
				in_right_window = False
				
			shift_right += 1


		# filling missing values
		if left_lane_found == False and right_lane_found == True:
			left_lane_pixel = np.int32(right_lane_pixel - self.lane_width)
			flag_lane_found = True
			
		elif left_lane_found == True and right_lane_found == False:
			right_lane_pixel = np.int32(left_lane_pixel + self.lane_width)
			flag_lane_found = True

		elif left_lane_found == False and right_lane_found == False:
			left_lane_pixel = np.int32(self.previous_center - self.lane_half_width) # calculated value when nothing found
			right_lane_pixel = np.int32(self.previous_center + self.lane_half_width)# calculated value when nothing found
			flag_lane_found = False
		
		elif left_lane_found == True and right_lane_found == True:
			flag_lane_found = True
			self.lane_width = right_lane_pixel - left_lane_pixel
			self.lane_half_width = np.int32(self.lane_width / 2)
		
		
		lane_width_measure = right_lane_pixel - left_lane_pixel
		lane_width_measure_m = lane_width_measure*self.pixel2meter
		print("lane_width", lane_width_measure)
		print("lane_width_m", lane_width_measure_m)

		lane_center = left_lane_pixel + (right_lane_pixel - left_lane_pixel) // 2

		self.previous_left = left_lane_pixel
		self.previous_right = right_lane_pixel
		self.previous_center = lane_center

		# calculating the lateral displacement with moving average		
		lateral_displacement = ((lane_center - self.car_center_img) + self.lateral_displacement_pre + self.lateral_displacement_ppre)/3

		# writing update values for moving average
		self.lateral_displacement_ppre = self.lateral_displacement_pre
		self.lateral_displacement_pre = lateral_displacement

		# set the variable to Float32 so it is suitible for the ros topic
		lateral_displacement_m = Float32()
				
		# getting the lateral displacement in meter	and write the data 	
		lateral_displacement_m.data = lateral_displacement*self.pixel2meter

		# publishing the displacement
		self.pub_2_topic_lat_displacement(lateral_displacement_m)	
		
		#timer
		if self.stopwatch_flag:		
			stop_fastlanedetct = time.time()
			time_fastlanedetct = stop_fastlanedetct - start_fastlanedetct
			print("time_fastlanedetct ->", time_fastlanedetct)
						
		# visualisation
		if self.develop_flag:

			# scanrow_bgr_HDwebCam
			self.image_pub21.publish(self.bridge.cv2_to_imgmsg(scanrow_img, "bgr8"))
		
			# convert gray to color for visualisation (8uc1 to 8uc3)
			blur_pub = cv.cvtColor(blur_aoi, cv.COLOR_GRAY2RGB) #convert 8uc1 into 8uc3

			# scanrow_bw_HDwebCam
			blur_windows = blur_pub.copy()
			blur_windows = cv.rectangle(blur_windows,(window_left_begin,self.sum_hight-1),(window_left_end,0),self.color_blue,1) #left window
			blur_windows = cv.rectangle(blur_windows,(window_right_begin,self.sum_hight-1),(window_right_end-1,0),self.color_blue,1) #right window
			self.image_pub23.publish(self.bridge.cv2_to_imgmsg(blur_windows, "bgr8"))

			# lane_position_HDwebCam
			blur_pub = cv.line(blur_pub, (default_center, self.sum_hight-1),(default_center, 0),self.color_purple,3) # center of car in rect aoi
			blur_pub = cv.line(blur_pub, (lane_center, self.sum_hight-1),(lane_center, 0),self.color_red,1) # lanecenter
			blur_pub = cv.line(blur_pub, (left_lane_pixel, self.sum_hight-1),(left_lane_pixel, 0),self.color_green,1) # left lane
			blur_pub = cv.line(blur_pub, (right_lane_pixel, self.sum_hight-1),(right_lane_pixel, 0),self.color_green,1) # right lane
			self.image_pub20.publish(self.bridge.cv2_to_imgmsg(blur_pub, "bgr8"))

			# calibration_image_HDwebCam
			cal_img_HDwebCam = cv.line(cv_image_orig, (self.x_dim_min, self.y_dim_center),(self.x_dim_max, self.y_dim_center),self.color_blue,1) # scanrow
			cal_img_HDwebCam = cv.line(cal_img_HDwebCam, (self.x_dim_min, self.y_dim_center-self.sum_hight),(self.x_dim_max, self.y_dim_center-self.sum_hight),self.color_blue,1) # scanrow
			cal_img_HDwebCam = cv.line(cal_img_HDwebCam, (self.x_dim_center, self.y_dim_max),(self.x_dim_center, self.y_dim_min),self.color_purple,1) # center of car in rect aoi
			cal_img_HDwebCam = cv.line(cal_img_HDwebCam, (left_lane_pixel,self.y_dim_center),(left_lane_pixel,self.y_dim_center-20),self.color_green,2) # left lane
			cal_img_HDwebCam = cv.line(cal_img_HDwebCam, (right_lane_pixel,self.y_dim_center),(right_lane_pixel,self.y_dim_center-20),self.color_green,2) # right lane
			cal_img_HDwebCam = cv.line(cal_img_HDwebCam, (right_lane_pixel,self.y_dim_center-10),(left_lane_pixel,self.y_dim_center-10),self.color_green,2) # measurelane
			cal_img_HDwebCam = cv.putText(cal_img_HDwebCam, "lane width: " + str(lane_width_measure) +" pixel",
						 				 np.int32((self.x_dim_max *0.25, self.y_dim_center-30)),
										 cv.FONT_HERSHEY_SIMPLEX, 1, 
										 self.color_green, 1, cv.LINE_AA)
		
			# image before hls black and white
			self.image_pub30.publish(self.bridge.cv2_to_imgmsg(cal_img_HDwebCam, "bgr8"))




	################################
	# PUBLISH LATERAL DISPLACEMENT #
	################################

	def pub_2_topic_lat_displacement(self, lateral_displacement_m):

		print("publisch lat disp")
		self.get_logger().info("publisch lat disp: %s" % str(lateral_displacement_m.data)) 
		try:			
			self.control_value.publish(lateral_displacement_m)
		except KeyError:
			pass

		
	


def main(args=None):
	rclpy.init(args=args)
	node = LaneDetection()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
