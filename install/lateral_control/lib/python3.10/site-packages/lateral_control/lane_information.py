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
from collections import Counter 
from rclpy.qos import qos_profile_sensor_data
from ackermann_msgs.msg import AckermannDriveStamped
import time
from time import localtime, strftime

class LaneDetection(Node):

	def __init__(self): 
		super().__init__('lane_information')
		
		# declare parameters
		self.declare_parameter('develop', 			False)	
		self.declare_parameter('stopwatch', 		False)
		self.declare_parameter('thresh_ZED', 		120)
		self.declare_parameter('resolution_ZED',	376)
		self.declare_parameter('horizon_dist', 		1)
		self.declare_parameter('color_white', 		(255, 255, 255))
		self.declare_parameter('color_red',			(0, 0, 255))
		self.declare_parameter('color_green',		(0, 255, 0))
		self.declare_parameter('color_blue',		(255, 0, 0))
		self.declare_parameter('color_purple',		(155, 25, 220))
		self.declare_parameter('color_gray',		(155, 25, 220))

		# retrieve parameters
		self.develop_flag 		= self.get_parameter('develop').value			
		self.stopwatch_flag 	= self.get_parameter('stopwatch').value
		self.thresh_ZED			= self.get_parameter('thresh_ZED').value
		self.resolution_ZED		= self.get_parameter('resolution_ZED').value
		self.horizon_dist		= self.get_parameter('horizon_dist').value
		self.color_white		= self.get_parameter('color_white').value
		self.color_red			= self.get_parameter('color_red').value
		self.color_green		= self.get_parameter('color_green').value
		self.color_blue			= self.get_parameter('color_blue').value
		self.color_purple		= self.get_parameter('color_purple').value
		self.color_gray			= self.get_parameter('color_gray').value

		self.get_logger().info("parameter 'resolution_ZED' has value: " + str(self.resolution_ZED))
		self.get_logger().info("parameter 'horizon_dist' has value: " + str(self.horizon_dist))
		
		self.bridge = CvBridge()

		# Subcribers
		self.image_sub = self.create_subscription(Image, '/ZEDcam/image_raw', self.callback_lane_information, 10) #/zed/zed_node/stereo_raw/image_raw_color
		self.lat_disp_sub = self.create_subscription(Float32, "lateral_displacement", self.callback_lat_disp_sub, qos_profile_sensor_data)
		self.latest_lat_disp = None
 
		# Publishers
		self.lane_info = self.create_publisher(Int8MultiArray, "lane_info", 100)
		
		self.image_pub2 = self.create_publisher(Image, "dashboard", 10)
		self.curve_timo_publisher = self.create_publisher(Float32, 'steering_speed_pca' , 10)

		# video streams for function insights
		if self.develop_flag:	
			self.image_pub20 = self.create_publisher(Image,"calibration_image_ZEDcam",10)
			self.image_pub21 = self.create_publisher(Image, "rect_FoI_ZEDcam", 10)
			self.image_pub22 = self.create_publisher(Image, "lane_detection_ZEDcam", 10)

		
		# Save times
		if self.stopwatch_flag:
			self.csv_file_time  = open('lane_information_time.csv', 'w')
			self.csv_writer_time    = csv.writer(self.csv_file_time)
			self.csv_writer_time.writerow(['time_calc_params','time_cropimage','time_wrapPerspective','time_cv_brg2hls','time_thresh',
											'time_blur','time_birdview','time_lr_scan','time_post_processing',
											'time_calc_lines','time_col_to_gray','time_cv_drawings','time_pub_bird_lane',
											'time_visual_bird', 'time_dashboard','time_callback_lane_information'])
	
		# initialising Tokens
		self.init_calc_did_run = False
		self.lane_is_detected = True
			
	
	def __del__(self):
		self.csv_file_time.close()

	# save the latest lateral displacement from the in_lane_postitioning node
	def callback_lat_disp_sub(self,disp):
		self.latest_lat_disp = 0.0 #disp.data 


	def callback_lane_information(self,data):
		time1 = time.time()

		print("--------------------------------------------------")	
		print("NEW ITERATION")
		print("--------------------------------------------------")

		#timer
		start_callback_lane_information = time.time()

		self.t_past = int(round(time.time()*1000))	#time in ms getrs round and converted in sec	
		try:
			cv_image_orig = self.bridge.imgmsg_to_cv2(data, "bgr8") #camera picture is converted to the cv2 type so that in can be processed by OpenCV functions
		except CvBridgeError as e: 
			print(e) #if there is an error it will be printed out

		# # Flipping the image around the y-axis (only for the rosbag)
		# cv_image_orig = cv.flip(cv_image_orig, 1) 

		#timer
		if self.stopwatch_flag:
			start_calc_params = time.time()

		#Define area and positions for lane detection
		#marcel: only once for performance reasons provided image will not change while running code
		if self.init_calc_did_run == False:

			# get the dimensions of the ZED Image from the cv_image_orig
			self.get_dimenson_of_ZED_image(cv_image_orig)

			# calculating the dimensions for the foi
			self.get_dimenson_foi()
								
			# get the points for the trapezoid
			self.calc_points_trapezoid()

			#calculating the convertion factor from Pixel to Meter and reverse
			self.calc_pixel2meter()
		
			# calculating the car center including the camera offset
			self.calc_car_center_in_images()

			# calculate parameters for the rectification of the image
			self.calc_rect_params()
			#self.calc_shrunk_rect_params()
		
			# matrix for the scanrows
			self.initialize_matrix_scanrows()

			# matrix for info about the lane
			self.initialize_info_matrix()
											
			print("lane_matrix dimnensons___", self.number_of_scanrows)
			print("car_center_img_left:_____", self.car_center_img_left)
			print("car_center_img_right:____", self.car_center_img_right)
			print("y_dim_min:_______________", self.y_dim_min)
			print("y_dim_max:_______________", self.y_dim_max)	
			print("x_dim_min:_______________", self.x_dim_min)
			print("x_dim_center:____________", self.x_dim_center)
			print("x_dim_max:_______________", self.x_dim_max)
			print("center of left image:____", self.img_left_center)
			print("y_foi_horizon_1000mm:____", self.y_foi_horizon_1000mm)
			print("y_foi_horizon_2000mm:____", self.y_foi_horizon_2000mm)
			print("y_dim_foi_2000mm:________", self.y_dim_foi_2000mm)
			print("y_dim_foi_1000mm:________", self.y_dim_foi_1000mm)
			print("x_dim_foi:_______________", self.x_dim_foi)				
			print("image measures initialised")	
			self.init_calc_did_run = True

			# timer
			if self.stopwatch_flag:
				stop_calc_params = time.time()
				self.time_calc_params = stop_calc_params - start_calc_params

		# image processing		
		blur_aoi = self.image_rectification(cv_image_orig) 

		# lane calculation
		lane_matrix, flag_dashed = self.lane_calculation_dashed(blur_aoi) 
		
		# neighbor lane calculation
		if flag_dashed in (-1, 1):
			lane_matrix = self.neighbor_lane_calculation(blur_aoi, lane_matrix) 

		# dasboard view 			
		self.dashboard_view(cv_image_orig, blur_aoi, lane_matrix)

		# video streams for developent purposes
		if self.develop_flag:
			self.measure_trapezoid(cv_image_orig)
			self.make_rect_FoI_ZEDcam(lane_matrix, blur_aoi)
			self.visualisation_of_birdview(flag_dashed, lane_matrix, blur_aoi)
			 				
		##################################################################################
	
	
		if self.stopwatch_flag:
			# timer
			stop_callback_lane_information = time.time()
			self.time_callback_lane_information = stop_callback_lane_information - start_callback_lane_information
			print("time_callback_lane_information ->", self.time_callback_lane_information)
			self.csv_writer_time.writerow([self.time_calc_params,self.time_cropimage,self.time_wrapPerspective,self.time_cv_brg2hls,self.time_thresh,
								  			self.time_blur,self.time_birdview,self.time_lr_scan,self.time_post_processing,
											self.time_calc_lines,self.time_col_to_gray,self.time_cv_drawings,self.time_pub_bird_lane,
											self.time_visual_bird, self.time_dashboard, self.time_callback_lane_information])
			self.csv_file_time.flush()

	# ALL SUBORDINATE FUNCTIONS IN THE FOLLOWING #
	
	#################################################
	# READING THE DIMENSIONS OF THE ZED INPUT IMAGE #
	#################################################

	def get_dimenson_of_ZED_image(self, cv_image_orig):
		# Dimensions of the original ZED Image
		self.y_dim_min = 0
		self.y_dim_max = cv_image_orig.shape[0] #376p up to 1242p
		self.x_dim_min = 0
		self.x_dim_center = cv_image_orig.shape[1] // 2 #maybe set to a certain Pixel length
		self.x_dim_max = cv_image_orig.shape[1]
		self.img_left_center = (self.x_dim_center-self.x_dim_min) // 2 # center of the left image


	#########################################
	# CALCULATING THE DIMENSIONS OF THE FOI #
	#########################################

	def get_dimenson_foi(self):
		self.x_dim_foi = self.x_dim_center			

		# Horizon in the image for the lane detection	
		if self.resolution_ZED == 376:
			self.y_foi_horizon_1000mm = np.int32(self.y_dim_max / 1000 *644) #horizon in pixels for the frame of interest
			self.y_foi_horizon_2000mm = np.int32(self.y_dim_max / 1000 * 578) #for 2000 mm

		elif self.resolution_ZED == 1242:
			self.y_foi_horizon_1000mm = np.int32(self.y_dim_max / 1000 *650) #horizon in pixels for the frame of interest
			self.y_foi_horizon_2000mm = np.int32(self.y_dim_max / 1000 * 578) #for 2000 mm

		else:
			self.get_logger().error("parameter 'resolution_ZED' has invalid value: " + str(self.resolution_ZED))

		self.y_dim_foi_1000mm = self.y_dim_max - self.y_foi_horizon_1000mm
		self.y_dim_foi_2000mm = self.y_dim_max - self.y_foi_horizon_2000mm
	

	#####################################
	# MEASURED POINTS FOR THE TRAPEZOID #
	#####################################

	def calc_points_trapezoid(self):
		if self.horizon_dist == 1:
			self.y_foi_horizon = self.y_foi_horizon_1000mm # from the top 0 down to the pixel value of the horizon
			self.y_dim_foi = self.y_dim_foi_1000mm # height of the foi in pixel

			if self.resolution_ZED == 376:
				self.point_aoi_tl = 245 #765 #227 367p measured
				self.point_aoi_tr = 425 #1449 #443 376p measured
				self.firstrow_width = 0.755 #720 #890 # 870 376p [m]
				self.horizon_width = 2.830 #2.970 #4000 #2280 #[m] 2560 measured
			elif self.resolution_ZED == 1242:
				self.point_aoi_tl = 765 #227 367p measured
				self.point_aoi_tr = 1449 #443 376p measured
				self.firstrow_width = 0.720 # [m]
				self.horizon_width = 4.400 #2560 measured
				
			else:
				self.get_logger().error("parameter 'resolution_ZED' has invalid value: " + str(self.resolution_ZED))
				
			print("Horizon of detecton is 1000 mm away")

		elif self.horizon_dist == 2:
			self.y_foi_horizon = self.y_foi_horizon_2000mm
			self.y_dim_foi = self.y_dim_foi_2000mm
			
			if self.resolution_ZED == 376:
				self.point_aoi_tl = 283 #915 #901->1242p #915->376p
				self.point_aoi_tr = 389 #1280 # 1312->1242p #1280->376p
				self.firstrow_width = 0.755 #0.890 #376p 
				self.horizon_width = 4.800
			elif self.resolution_ZED == 1242:
				self.point_aoi_tl = 915 #901->1242p #915->376p
				self.point_aoi_tr = 1280 # 1312->1242p #1280->376p
				self.firstrow_width = 0.720 # [m]
				self.horizon_width = 4.400
			else:
				self.get_logger().error("parameter 'resolution_ZED' has invalid value: " + str(self.resolution_ZED))
				
			print("Horizon of detecton is 2000 mm away")
		
		else:
			self.get_logger().error("parameter 'horizon_dist' has invalid value: " + str(self.horizon_dist))


	#####################################################################
	# CALCULATING THE CONVERSION FACTOR FROM PIXEL TO METER AND REVERSE #
	#####################################################################	
		
	def calc_pixel2meter(self):
		# calculating from pixel to meter 
		self.pixel2meter = self.firstrow_width / self.x_dim_center # ZED camera:meters per pixel; 870 mm width measured at 35cm distance at the first visible pixle row for 376p /// 0.49/640 #Logitech camera ///
		self.meter2pixel = 1 / self.pixel2meter


	##############################################################
	# CALCULATING THE X-POSITION OF THE CAR CENTER IN THE IMAGES #
	##############################################################

	def calc_car_center_in_images(self):
		
		self.offset_lense = 0.060 #0.06 #ZED camera horizontal offset from center of front axis, +6cm for left lens
		self.car_center_img_left = np.int32((self.x_dim_center / 2) + (self.offset_lense * self.meter2pixel)) # half of the image plus the offset of the lens (60 mm)
		self.car_center_img_right = np.int32((self.x_dim_center / 2) - (self.offset_lense * self.meter2pixel)) # half of the image plus the offset of the lens (60 mm)


	##############################################
	# CALCULATING MEASUREMENTS FOR RECTIFICATION #
	##############################################

	def calc_rect_params(self):

		
		# perspective transformation values for 2000 mm, 30 pixels foresight or two modules
		self.points_aoi = np.float32([
			[self.point_aoi_tl,				 0],	#topleft	[278, 0] values from experiments --- old values912/2208
			[self.point_aoi_tr,				 0],	#topright	[397, 0] values from experiments --- old values 1308/2208
			[self.x_dim_foi, 	self.y_dim_foi],	#bottomright [672, 158]
			[0,					self.y_dim_foi],	#bottomleft  [0,   158]
		])
					
		# # desired corner points of the rectifyed aoi
		#ratio of perspective with to length 2000 mm : 740 mm 
		ratio_w_l = np.float32(self.horizon_dist/self.firstrow_width) #900

		# ratio of the with in the front and the horizon of the aoi
		ratio_width = np.float32(self.horizon_width/self.firstrow_width) #measured 4180/720
				
		# width and height of the rectified aoi
		self.width_rect_aoi = np.int32(self.x_dim_foi * ratio_width)
		self.height_rect_aoi = np.int32(self.x_dim_foi * ratio_w_l) #np.int32(672//ratio)
		
		# location of the bottom corners of the camera frame in the rect aoi
		self.corner_left_in_rect_aoi = np.int32(((self.x_dim_foi * ratio_width)-self.x_dim_foi)/2)
		self.corner_right_in_rect_aoi = np.int32(((self.x_dim_foi * ratio_width)+self.x_dim_foi)/2)
		print("left and right corner in the aoi", self.corner_left_in_rect_aoi, self.corner_right_in_rect_aoi)

		# location of the car_center in the rect aoi
		self.car_center_rect_aoi_left = np.int32((self.x_dim_foi * ratio_width / 2) - (self.x_dim_foi / 2)  + self.car_center_img_left)#*1.05)
		self.car_center_rect_aoi_right = np.int32((self.x_dim_foi * ratio_width / 2) - (self.x_dim_foi / 2)  + self.car_center_img_right)#*0.95)

		print("car_center_rect_aoi_left", self.car_center_rect_aoi_left)
		print("car_center_rect_aoi_right", self.car_center_rect_aoi_right)
							
		# Frame where the aoi should fit 
		self.points_rect_aoi = np.float32([
			[self.corner_left_in_rect_aoi,						0],		#topleft		[0,0]
			[self.corner_right_in_rect_aoi,						0],		#topright		[672,0]
			[self.corner_right_in_rect_aoi,	self.height_rect_aoi],		#bottomright	[672,1634]
			[self.corner_left_in_rect_aoi,	self.height_rect_aoi],		#bottomleft		[0, 1634]
		])

		# matrix for transformation for the rectification
		self.rect_trans_matrix = cv.getPerspectiveTransform(self.points_aoi,self.points_rect_aoi)
		self.inv_rect_trans_matrix = cv.getPerspectiveTransform(self.points_rect_aoi, self.points_aoi)

		

	#####################################################
	# CALCULATING SHRUNK MEASUREMENTS FOR RECTIFICATION #
	#####################################################
		
	def calc_shrunk_rect_params(self):			
		#timer
		start_calc_points_rect = time.time()
							
		#timer
		start_calc_points_rect = time.time()

		self.shrunk_divisor = 10
		self.x_dim_foi = np.int32(self.x_dim_foi /self.shrunk_divisor)
		self.y_dim_foi = np.int32(self.y_dim_foi /self.shrunk_divisor)
		self.pixel2meter = self.pixel2meter *self.shrunk_divisor
		self.car_center_img_left = np.int32(self.car_center_img_left /self.shrunk_divisor)
		self.car_center_img_right = np.int32(self.car_center_img_right /self.shrunk_divisor)

		# perspective transformation values for 2000 mm, 30 pixels foresight or two modules
		self.points_aoi = np.float32([
			[(self.point_aoi_tl/2208)*self.x_dim_foi,				0],	#topleft	[278, 0] values from experiments --- old values912/2208
			[(self.point_aoi_tr/2208)*self.x_dim_foi,				0],	#topright	[397, 0] values from experiments --- old values 1308/2208
			[self.x_dim_foi, 							self.y_dim_foi],	#bottomright [672, 158]
			[0,		 									self.y_dim_foi],	#bottomleft  [0,   158]
		])
				
		# # desired corner points of the rectifyed aoi
		#ratio of perspective with to length 2000 mm : 740 mm 
		ratio_w_l = np.float32(self.horizon_dist/self.firstrow_width) #900

		# ratio of the with in the front and the horizon of the aoi
		ratio_width = np.float32(self.horizon_width/self.firstrow_width) #measured 4180/720
				
		# width and height of the shrunk rectified aoi
		self.width_rect_aoi = np.int32(self.x_dim_foi * ratio_width)
		self.height_rect_aoi = np.int32(self.x_dim_foi * ratio_w_l) #np.int32(672//ratio)
		print("width and height of the rect aoi:", self.width_rect_aoi, self.height_rect_aoi)
		
		# location of the bottom corners of the camera frame in the rect aoi
		self.corner_left_in_rect_aoi = np.int32(((self.x_dim_foi * ratio_width)-self.x_dim_foi)/2)
		self.corner_right_in_rect_aoi = np.int32(((self.x_dim_foi * ratio_width)+self.x_dim_foi)/2)
		print("left and right corner in the aoi", self.corner_left_in_rect_aoi, self.corner_right_in_rect_aoi)

		# location of the car_center in the rect aoi
		self.car_center_rect_aoi_left = np.int32((self.x_dim_foi * ratio_width / 2) - (self.x_dim_foi / 2)  + self.car_center_img_left*1.05)
		self.car_center_rect_aoi_right = np.int32((self.x_dim_foi * ratio_width / 2) - (self.x_dim_foi / 2)  + self.car_center_img_right*0.95)

		print("car_center_rect_aoi_left", self.car_center_rect_aoi_left)
		print("car_center_rect_aoi_right", self.car_center_rect_aoi_right)
							
		# Frame where the aoi should fit 
		self.points_rect_aoi = np.float32([
			[self.corner_left_in_rect_aoi,						0],		#topleft		[0,0]
			[self.corner_right_in_rect_aoi,						0],		#topright		[672,0]
			[self.corner_right_in_rect_aoi,	self.height_rect_aoi],		#bottomright	[672,1634]
			[self.corner_left_in_rect_aoi,	self.height_rect_aoi],		#bottomleft		[0, 1634]
		])

		# matrix for transformation for the rectification
		self.rect_trans_matrix = cv.getPerspectiveTransform(self.points_aoi,self.points_rect_aoi)
		self.inv_rect_trans_matrix = cv.getPerspectiveTransform(self.points_rect_aoi, self.points_aoi)

		# timer
		stop_calc_points_rect = time.time()
		self.time_calc_points_rect = stop_calc_points_rect - start_calc_points_rect




	##############################################
	# INTITIALISING THE MATRIX FOR THE SCNANROWS #
	##############################################

	def initialize_matrix_scanrows(self):
			
		self.number_of_sections = np.int32(self.horizon_dist *10)
		self.number_of_scanrows = 10
		self.initial_lane_matrix = np.zeros((self.number_of_scanrows,17),dtype=int)




	##############################################
	# INTITIALISING THE MATRIX FOR VISUALITATION #
	##############################################

	def initialize_info_matrix(self):
		self.info_matrix = np.zeros((6,3),dtype=int)



	
	#############################
	# Calculating the birdsview #
	#############################

	# compressung the image approach
	def image_rectification(self, cv_image_orig):
		
		#timer
		if self.stopwatch_flag:
			start_birdview = time.time()
				
		# Frame of interest of the left zed imgage
		cv_image_left_foi = cv_image_orig[self.y_foi_horizon:self.y_dim_max, self.x_dim_min:self.x_dim_foi] #cropded frame of picture for the lanedetection

		if self.stopwatch_flag:		
			stop_cropimage = time.time()
			self.time_cropimage = stop_cropimage - start_birdview
			
		## shrinking
		# cv_image_left_foi = cv.resize(cv_image_left_foi, (self.x_dim_foi, self.y_dim_foi)) # image with (width and height)

		# timer
		if self.stopwatch_flag:
			start_wrapPerspective = time.time()

		# image transformation			image,			trans matrix			(width, height)
		rect_aoi = cv.warpPerspective(cv_image_left_foi, self.rect_trans_matrix, (self.width_rect_aoi, self.height_rect_aoi))#, flags=(cv.INTER_LINEAR))

		#timer	
		if self.stopwatch_flag:		
			stop_wrapPerspective = time.time()
			self.time_wrapPerspective = stop_wrapPerspective - start_wrapPerspective
		
		#timer
		if self.stopwatch_flag:
			start_cv_brg2hls = time.time()
		
		# gbr to hls convertion
		rect_aoi_hls = cv.cvtColor(rect_aoi, cv.COLOR_BGR2HLS) #gives every bixeltransformiert das RGB Bild in Grauwerte von 0 bis 188 0:hue, farbe 1:grayValus, 2:saturation sättigung 
		
		#timer
		if self.stopwatch_flag:
			stop_cv_brg2hls = time.time()
			self.time_cv_brg2hls = stop_cv_brg2hls - start_cv_brg2hls
			
		#timer
		if self.stopwatch_flag:
			start_thresh = time.time()

		# theashold to get a black and white image
		_ , rect_aoi_hls_threshold = cv.threshold(rect_aoi_hls[:,:,1], self.thresh_ZED, 255, cv.THRESH_BINARY) #original(100,255) #dimensions [y_pixel,x_pixel,informations], sets a threshold on the grayvalues to make the picture black and white
		
		#timer	
		if self.stopwatch_flag:		
			stop_thresh = time.time()
			self.time_thresh = stop_thresh - start_thresh
						
		#timer
		if self.stopwatch_flag:
			start_blur = time.time()
		
		# blur image
		blur_aoi = cv.GaussianBlur(rect_aoi_hls_threshold, (3,3), 0) #blurring the black and white picture

		#timer	
		if self.stopwatch_flag:		
			stop_blur = time.time()
			self.time_blur = stop_blur - start_blur
			
		#timer
		if self.stopwatch_flag:		
			stop_birdview = time.time()
			self.time_birdview = stop_birdview - start_birdview

		return blur_aoi

	
	

	#####################################################################################
	# CALC POSITION OF LANE LINES WITH THE HELP OF THE PREVIOUS LANES # DASHED POSSIBLE #
	#####################################################################################

	def lane_calculation_dashed(self, blur_aoi):

		# timer
		if self.stopwatch_flag:
			start_calc_lines = time.time()
			
		# import lane_matrix as a local variable
		lane_matrix = self.initial_lane_matrix
				
		# getting the startingpoint for searching
		initial_car_center = self.car_center_rect_aoi_left

		# getting the lane with
		lane_half_width = np.int32(self.x_dim_foi/(724/175)) # the width two lanes with lines 724 mm fits in the wirst camera row. 175 mm is half of the lane width
				
		course = 0
		scanrow = 0	
		scanrows = self.number_of_scanrows
		while scanrow < scanrows : 

			#print(" ")
			#print("scanrow", scanrow,"out of", scanrows-1)
						
			# deciding which lane to scan
			scanrow_pixelpos= np.int32(self.height_rect_aoi - (self.height_rect_aoi / self.number_of_sections * scanrow))

			# calculating the default center and the search windows for the next lane
			default_center, window_left_begin, window_left_end, window_right_begin, window_right_end = self.calc_default_center_and_windows(initial_car_center, lane_half_width, scanrow, scanrow_pixelpos, lane_matrix)
			
			# finding the lane lines in the current scanrow
			left_lane_found, right_lane_found, left_lane_pixel, right_lane_pixel = self.find_lane_lines(scanrow_pixelpos, window_left_begin, window_left_end, window_right_begin, window_right_end, blur_aoi)

			# get default values if one or both lanes are not found
			left_lane_pixel, right_lane_pixel = self.fill_line_gaps(lane_half_width, scanrow, left_lane_found, right_lane_found, left_lane_pixel, right_lane_pixel, default_center, lane_matrix)
									
			# filling the matrix with the values
			lane_matrix[scanrow,0] = scanrow_pixelpos
			lane_matrix[scanrow,1] = left_lane_pixel
			lane_matrix[scanrow,2] = left_lane_pixel + (right_lane_pixel - left_lane_pixel) // 2
			lane_matrix[scanrow,3] = right_lane_pixel 
			lane_matrix[scanrow,4] = window_left_begin
			lane_matrix[scanrow,5] = window_left_end
			lane_matrix[scanrow,6] = window_right_begin
			lane_matrix[scanrow,7] = window_right_end
			lane_matrix[scanrow,8] = left_lane_found
			lane_matrix[scanrow,9] = right_lane_found
							
			# count up to the next scanrow
			scanrow += 1

		#timer
		if self.stopwatch_flag:
			stop_lr_scan = time.time()
			self.time_lr_scan = stop_lr_scan - start_calc_lines
			#print("time_lr_scan", self.time_lr_scan)
		
		#timer
		if self.stopwatch_flag:
			start_post_processing = time.time()

		# calculating the width of the lane
		lane_calc_width = self.calc_lane_width(lane_matrix)

		#calculating the lateral displacement
		lateral_displacement = self.calc_lateral_displacement()

		# check for valid lane and lane condition
		flag_lane_found, flag_dashed = self.check_lane_contidition(lane_matrix, lane_calc_width)
			
		# smooth the upper halft of the lane line values
		lane_matrix = self.smooth_values(scanrows, flag_lane_found, lane_matrix)

		# Struckture of the info_matrix:
		#[[first center x,	first center y, lateral_displacement],
		# [last center x,	last center y,				  radius],
		# [curve center x,	curve center y, 			  course],
		# [0,				0,					 flag_lane_found],
		# [0,				0,						 flag_dashed]]

		#writing information about te first and the last centerpoint into info_matrix 
		self.info_matrix[0,0] = lane_matrix[0, 2] # first center x value
		self.info_matrix[0,1] = lane_matrix[0, 0] # first center y value
		self.info_matrix[1,0] = lane_matrix[scanrows-1, 2] # last center x value
		self.info_matrix[1,1] = lane_matrix[scanrows-1, 0] # last center y value
		
		# calculating the radius of the curve
		radius, radius_m = self.calc_radius(scanrows, lane_matrix)
		radius2 = Float32()
		radius2.data = radius_m
		#print("~~~~~~~Type of radius_m is: ", type(radius_m))
		self.curve_timo_publisher.publish(radius2)

		# deciding the course of the lane
		course = self.course_of_lane(radius_m)
		
		# filling the matrix with the calculatet values
		self.info_matrix[2,0] = lane_matrix[0, 2] - radius # curve center point x-value
		self.info_matrix[2,1] = lane_matrix[0, 0] # curve center point y-value
		self.info_matrix[0,2] = lateral_displacement
		self.info_matrix[1,2] = radius
		self.info_matrix[2,2] = course
		self.info_matrix[3,2] = flag_lane_found
		self.info_matrix[4,2] = lane_calc_width
		self.info_matrix[5,2] = flag_dashed
		

		if self.develop_flag:
			print("lane_matrix")
			print(lane_matrix)
			print("info_matrix")
			print(self.info_matrix)
			print("lane_calc_width", lane_calc_width*self.pixel2meter)

		if self.stopwatch_flag:
			stop_post_processing = time.time()
			self.time_post_processing = stop_post_processing - start_post_processing
				
		self.lane_info_publischer(flag_lane_found, course)

		# timer
		if self.stopwatch_flag:
			stop_calc_lines = time.time()
			self.time_calc_lines = stop_calc_lines - start_calc_lines

		print("laneMatrix after calculations", lane_matrix)
		print("initial_lane_matrix", self.initial_lane_matrix)
			
		return lane_matrix, flag_dashed
	



	###########################################################
	# CALC NEIGHBOR LANE WITH THE INFORMATION OF THE EGO LANE #
	###########################################################

	def neighbor_lane_calculation(self, blur_aoi, lane_matrix):

		
		neighbor_window_half_width = np.int32(self.x_dim_foi/16)
		scanrow = 0	
		scanrows = self.number_of_scanrows
		while scanrow < scanrows : 
			
			main_lane_width = lane_matrix[scanrow, 3] - lane_matrix[scanrow, 1]

			# deciding which lane to scan
			scanrow_pixelpos= np.int32(self.height_rect_aoi - (self.height_rect_aoi / self.number_of_sections * scanrow))

			# calculating the default center and the search windows for the next lane
			default_center, window_left_begin, window_left_end, window_right_begin, window_right_end = self.calc_windows_for_neighbor_lane(main_lane_width, neighbor_window_half_width, scanrow, lane_matrix)
			
			# finding the lane lines in the current scanrow
			left_lane_found, right_lane_found, left_lane_pixel, right_lane_pixel = self.find_lane_lines(scanrow_pixelpos, window_left_begin, window_left_end, window_right_begin, window_right_end, blur_aoi)

			# get default values if one or both lanes are not found
			left_lane_pixel, right_lane_pixel = self.fill_line_gaps_neighbor_lane(main_lane_width , default_center, left_lane_found, right_lane_found, left_lane_pixel, right_lane_pixel)
		
			# filling the lane_matrix with the values
			lane_matrix[scanrow, 10] = left_lane_pixel
			lane_matrix[scanrow, 11] = left_lane_pixel + (right_lane_pixel - left_lane_pixel) // 2
			lane_matrix[scanrow, 12] = right_lane_pixel
			lane_matrix[scanrow, 13] = window_left_begin
			lane_matrix[scanrow, 14] = window_left_end
			lane_matrix[scanrow, 15] = window_right_begin
			lane_matrix[scanrow, 16] = window_right_end
										
			# count up to the next scanrow
			scanrow += 1

			
		# smooth the upper halft of the lane line values
		lane_matrix = self.smooth_neighbor_values(scanrows, lane_matrix)
					
		return lane_matrix



	##############################################################################
	# calculating the default center for the next scanrow and the search windows #
	##############################################################################

	def calc_default_center_and_windows(self, initial_car_center, lane_half_width, scanrow, scanrow_pixelpos, lane_matrix):

		# the scanrow 0 is searching the lanes from the carcenter to the outside
		if scanrow == 0:
			# define default center
			default_center = initial_car_center

			# scan window for left lane
			window_left_begin = 	default_center - lane_half_width*2
			window_left_end = 		default_center
		
			# scan window for right lane 
			window_right_begin = 	default_center
			window_right_end = 		default_center + lane_half_width*2
					
		# searching the second row with previous value as default value
		elif scanrow == 1 or scanrow == 2:
			# center lane
			default_center = 		np.int32(lane_matrix[scanrow-1,2])
									
			# scan window for left lane 
			window_left_begin = 	np.int32(default_center - lane_half_width * 1.8)
			window_left_end = 		np.int32(default_center - (lane_half_width /2))

			# scan window for right lane 
			window_right_begin = 	np.int32(default_center + (lane_half_width /2))
			window_right_end = 		np.int32(default_center + lane_half_width * 1.8)

		# searching for further lines beginning at the fourth scanrow 
		# missing value can be predicted with polinomial interpolation
		else: 
			# getting all the previous scanrow positions
			previous_pixel_lines = 	np.array(lane_matrix[:scanrow,0])
			
			# get previous information about the lane (last 3)
			previous_center_points = np.array(lane_matrix[:scanrow,2])

			# make coefficients for a quadratic polynimial (degree 2)
			coeff_center = np.polyfit(previous_pixel_lines,previous_center_points, 2)

			# Create a polynomial function with deh coefficients
			poly_center = np.poly1d(coeff_center)

			# Calculate the next left lane value value using the polynomial function
			default_center = np.int32(poly_center(scanrow_pixelpos))

			# check if value is valid
			if np.abs(default_center - lane_matrix[scanrow-1,2]) > (lane_half_width/2):
				default_center = lane_matrix[scanrow-1,2]

			prev_half_lane_width = np.abs((lane_matrix[scanrow-1,3] - lane_matrix[scanrow-1,1]) * 0.5)
		
			# scan window for left lane
			window_left_begin = np.int32(default_center - lane_half_width * 2)
			if window_left_begin < 0:
				window_left_begin = 50
			window_left_end = np.int32(default_center - (lane_half_width * 0.5))

			# scan window for right lane 
			window_right_begin = np.int32(default_center + (lane_half_width * 0.5))
			window_right_end = np.int32(default_center + lane_half_width * 2)
			if window_right_end > self.width_rect_aoi:
				window_right_end = self.width_rect_aoi - 50

		return default_center, window_left_begin, window_left_end, window_right_begin, window_right_end
	



	##############################################################################
	# calculating the default center for the next scanrow and the search windows #
	##############################################################################

	def calc_windows_for_neighbor_lane(self, main_lane_width, neighbor_window_half_width, scanrow, lane_matrix):

		# flag_dashed to know where to search for the neighbor lane
		flag_dashed	= self.info_matrix[5, 2]
		
		# the scanrow 0 is searching the lanes from the carcenter to the outside
		if flag_dashed == 1:

			main_left_line = lane_matrix[scanrow, 1]
			default_center =  lane_matrix[scanrow, 2] - main_lane_width
								
			# scan window for left lane
			window_left_begin = 	np.int32(main_left_line - main_lane_width - neighbor_window_half_width)
			window_left_end = 		np.int32(main_left_line - main_lane_width + neighbor_window_half_width)
		
			# scan window for right lane 
			window_right_begin = 	np.int32(main_left_line - neighbor_window_half_width)
			window_right_end = 		np.int32(main_left_line + neighbor_window_half_width)

		if flag_dashed == -1:

			main_right_line = lane_matrix[scanrow, 3]
			default_center =  lane_matrix[scanrow, 2] + main_lane_width
		
			# scan window for left lane
			window_left_begin = 	np.int32(main_right_line - neighbor_window_half_width)
			window_left_end = 		np.int32(main_right_line + neighbor_window_half_width)
		
			# scan window for right lane 
			window_right_begin = 	np.int32(main_right_line + main_lane_width - neighbor_window_half_width)
			window_right_end = 		np.int32(main_right_line + main_lane_width + neighbor_window_half_width)
	

		return default_center, window_left_begin, window_left_end, window_right_begin, window_right_end



	#######################################
	# scanning the scanrow for lane lines #
	#######################################
	
	def find_lane_lines(self, scanrow_pixelpos, window_left_begin, window_left_end, window_right_begin, window_right_end, blur_aoi):
	
		# initialising token for lane found status
		left_lane_found = 0 	
		right_lane_found = 0 	
		in_left_window = True
		in_right_window = True

		# initialising counter for for shifting the scanposition in the scanrow
		shift_left = 0 				
		shift_right = 0 			

		# left lane scan
		while left_lane_found == 0 and in_left_window == True:

			# the pixel index where the lane is searched					
			scan_index_left =  window_left_end - shift_left # the pixel index where the lane is searched

			# searching for white pixels in the window
			if scan_index_left >= window_left_begin and scan_index_left <= window_left_end:
									
				scan_left = blur_aoi[scanrow_pixelpos - 5:scanrow_pixelpos, scan_index_left - 1:scan_index_left]
				scan_left_sum = np.sum(scan_left, axis=0)
				if scan_left_sum >= 1000:
					left_lane_pixel = scan_index_left
					left_lane_found = 1
			else:
				in_left_window = False
				left_lane_pixel = 0

			shift_left += 1
		
		# right lane scan
		while right_lane_found == 0 and in_right_window == True:
			
			# the pixel index where the lane is searched					
			scan_index_right = window_right_begin + shift_right 
	
			# searching for white pixels in the window
			if scan_index_right >= window_right_begin and scan_index_right <= window_right_end:
				
				scan_right = blur_aoi[scanrow_pixelpos - 5:scanrow_pixelpos, scan_index_right - 1:scan_index_right]
				scan_right_sum = np.sum(scan_right, axis=0)
				if scan_right_sum >= 1000:
					right_lane_pixel = scan_index_right
					right_lane_found = 1
			else:
				in_right_window = False
				right_lane_pixel = 0
			
			shift_right += 1

		return left_lane_found, right_lane_found, left_lane_pixel, right_lane_pixel




	##################################################################
	# fill the gaps where nothing is detected with replacement value #
	##################################################################

	def fill_line_gaps(self, lane_half_width, scanrow, left_lane_found, right_lane_found, left_lane_pixel, right_lane_pixel, default_center, lane_matrix):

		# first iteration: no previous values
		if scanrow == 0:
			if left_lane_found == 0 and right_lane_found == 1:
				left_lane_pixel = np.int32(right_lane_pixel - lane_half_width*2)
				
			elif left_lane_found == 1 and right_lane_found == 0:
				right_lane_pixel = np.int32(left_lane_pixel + lane_half_width*2)

			elif left_lane_found == 0 and right_lane_found == 0:
				left_lane_pixel = np.int32(default_center - lane_half_width) # calculated value when nothing found
				right_lane_pixel = np.int32(default_center + lane_half_width)# calculated value when nothing found

		# second and third iteration: previous values available
		elif scanrow == 1 or scanrow == 2:
			if left_lane_found == 0 and right_lane_found == 1:
				left_lane_pixel = right_lane_pixel - lane_half_width*2
				
			elif left_lane_found == 1 and right_lane_found == 0:
				right_lane_pixel = left_lane_pixel + lane_half_width*2
				
			elif left_lane_found == 0 and right_lane_found == 0:
				left_lane_pixel = lane_matrix[scanrow-1,1] 
				right_lane_pixel = lane_matrix[scanrow-1,3] # taking the last laneposition 

		# fourth iteration: lanes can be predicted with a polynomial
		elif scanrow >= 3:
			prev_lane_half_width = np.int32(np.abs((lane_matrix[scanrow-1,3] - lane_matrix[scanrow-1,1]) * 0.5))

			if left_lane_found == 0 and right_lane_found == 1:
				left_lane_pixel = right_lane_pixel - prev_lane_half_width *2
				
			elif left_lane_found == 1 and right_lane_found == 0:
				right_lane_pixel = left_lane_pixel + prev_lane_half_width *2
				
			# calculated value when nothing found
			elif left_lane_found == 0 and right_lane_found == 0:
				left_lane_pixel = default_center - prev_lane_half_width *2
				right_lane_pixel = default_center + prev_lane_half_width *2
				
		return left_lane_pixel, right_lane_pixel





	##################################################################
	# fill the gaps where nothing is detected with replacement value #
	##################################################################

	def fill_line_gaps_neighbor_lane(self, main_lane_width , default_center, left_lane_found, right_lane_found, left_lane_pixel, right_lane_pixel):
		
		if left_lane_found == 0 and right_lane_found == 1:
			left_lane_pixel = np.int32(right_lane_pixel - main_lane_width)
			
		elif left_lane_found == 1 and right_lane_found == 0:
			right_lane_pixel = np.int32(left_lane_pixel + main_lane_width)

		elif left_lane_found == 0 and right_lane_found == 0:
			left_lane_pixel = np.int32(default_center - main_lane_width /2) # calculated value when nothing found
			right_lane_pixel = np.int32(default_center + main_lane_width /2)# calculated value when nothing found
	
				
		return left_lane_pixel, right_lane_pixel





	####################################
	# check if there is no lane at all #
	####################################
	
	
	def check_lane_contidition(self, lane_matrix, lane_calc_width):
		
		# with threshold
		width_min = 0.2 * self.meter2pixel
		width_max = 0.4 * self.meter2pixel

		# check if lane has valid width
		if width_min < lane_calc_width < width_max:

			line_flags = lane_matrix[:,8:10].copy()

			# Define the mapping from valid pairs to output values
			pair_to_value = {(0, 1): 1,		# left line missing
							(1, 0): -1, 	# right line missing
							(0, 0): 0, 	# no lines
							(1, 1): 2} 	# left and right line
			
			# store results
			scan_info = []

			# Check each row
			for row in line_flags:
				# Convert row to tuple
				row_tuple = tuple(row)
				# Get the corresponding output value
				value = pair_to_value.get(row_tuple, "Invalid pair")
				scan_info.append(value)

			print(scan_info)
					
			# Check if all elements in scan_info[:3] are not equal to 0
			first_three_valid = all(value != 0 for value in scan_info[:2])
			print("first three valid", first_three_valid)

			# Count occurrences of each value
			counts = Counter(scan_info)

			# decition what kind of lane
			if first_three_valid:
							
				if counts[2] >= 8 :
					flag_dashed = 0 # lane with solid lines
					flag_lane_found = 1

				elif counts[1] >= 3 and counts[2] >= 3:
					flag_dashed = 1 # lane with left line dashed
					flag_lane_found = 1

				elif counts[-1] >= 3 and counts[2] >= 3:
					flag_dashed = -1 # lane with right line dashed
					flag_lane_found = 1
				else:
					flag_lane_found = 0
					flag_dashed = 0
			else:
				flag_lane_found = 0
				flag_dashed = 0
		else:
			flag_lane_found = 0
			flag_dashed = 0
								
		return flag_lane_found, flag_dashed
	
	


	#################################################
	# smooth the upper half of the lane line values #
	#################################################

	def smooth_values(self, scanrows, flag_lane_found, lane_matrix):
		if flag_lane_found == 1:
			# smoothening of all the values
			#
			# getting all the previous scanrow positions and center positions
			all_pixel_lines =	np.array(lane_matrix[:,0])
			all_left_points =	np.array(lane_matrix[:,1])
			all_center_points = np.array(lane_matrix[:,2])
			all_right_points =	np.array(lane_matrix[:,3])

			# make coefficients for a quadratic polynimial (degree 2)
			coeff_left_all = 	np.polyfit(all_pixel_lines,all_left_points, 3)
			coeff_center_all = 	np.polyfit(all_pixel_lines,all_center_points, 3)
			coeff_right_all = 	np.polyfit(all_pixel_lines,all_right_points, 3)

			# Create a polynomial function with the coefficients
			poly_left_all =		np.poly1d(coeff_left_all)
			poly_center_all = 	np.poly1d(coeff_center_all)
			poly_right_all =	np.poly1d(coeff_right_all)

			smooth = np.int32(scanrows/2)
			while smooth < scanrows:
				# filling the matrix with the smoothed values
				lane_matrix[smooth,1] = np.int32(poly_left_all(all_pixel_lines[smooth]))
				lane_matrix[smooth,2] = np.int32(poly_center_all(all_pixel_lines[smooth]))
				lane_matrix[smooth,3] = np.int32(poly_right_all(all_pixel_lines[smooth]))
				
				smooth +=1
			
		else:
			# filling the matrix with default straight line
			lane_matrix[:,1] = np.int32(self.car_center_rect_aoi_left - self.x_dim_foi/4)
			lane_matrix[:,2] = self.car_center_rect_aoi_left
			lane_matrix[:,3] = np.int32(self.car_center_rect_aoi_left + self.x_dim_foi/4)
			
		return lane_matrix
	




	##########################################################
	# smooth the upper half of the neighbor lane line values #
	##########################################################

	def smooth_neighbor_values(self, scanrows, lane_matrix):
		
			# smoothening of all the values
			#
			# getting all the previous scanrow positions and center positions
			all_pixel_lines =	np.array(lane_matrix[:,0])
			all_left_points =	np.array(lane_matrix[:,10])
			all_center_points = np.array(lane_matrix[:,11])
			all_right_points =	np.array(lane_matrix[:,12])

			# make coefficients for a quadratic polynimial (degree 2)
			coeff_left_all = 	np.polyfit(all_pixel_lines,all_left_points, 3)
			coeff_center_all = 	np.polyfit(all_pixel_lines,all_center_points, 3)
			coeff_right_all = 	np.polyfit(all_pixel_lines,all_right_points, 3)

			# Create a polynomial function with the coefficients
			poly_left_all =		np.poly1d(coeff_left_all)
			poly_center_all = 	np.poly1d(coeff_center_all)
			poly_right_all =	np.poly1d(coeff_right_all)

			smooth = 1
			while smooth < scanrows:
				# filling the matrix with the smoothed values
				lane_matrix[smooth,10] = np.int32(poly_left_all(all_pixel_lines[smooth]))
				lane_matrix[smooth,11] = np.int32(poly_center_all(all_pixel_lines[smooth]))
				lane_matrix[smooth,12] = np.int32(poly_right_all(all_pixel_lines[smooth]))
				
				smooth +=1
			
			return lane_matrix




	#######################################
	# calculating the radius of the curve #
	#######################################

	def calc_radius(self, scanrows, lane_matrix):
		
		# values from the first and last centerlinepoint
		first_center_x = 	self.info_matrix[0,0]	
		first_center_y = 	self.info_matrix[0,1]	
		last_center_x = 	self.info_matrix[1,0]
		last_center_y = 	self.info_matrix[1,1]
				
		h = first_center_y - last_center_y
		off = first_center_x - last_center_x
		gamma = np.arctan(h/off)
		alpha = np.pi - gamma*2
		radius = h/np.sin(alpha)

		# limit max radius
		if radius > 200000:
			radius = 1000*self.x_dim_foi

		radius_m = radius * self.pixel2meter

		if self.develop_flag:
			print("radius", radius, "pixel")
			print("radius_m", radius_m)

		return radius, radius_m




	########################################
	# calculating the lateral displacement #
	########################################

	def calc_lateral_displacement(self):

		if self.latest_lat_disp is not None:
			lateral_displacement = self.latest_lat_disp * self.meter2pixel
			self.latest_lat_disp = None
		else:
			lateral_displacement = 0

		return lateral_displacement
	



	###################################
	# deciding the course of the lane #
	###################################

	def course_of_lane(self, radius_m):
			
		if np.absolute(radius_m) >= 3:
			course = 0
		elif radius_m > 0:
			course = 1
		elif radius_m <= 0:
			course = -1
			
		return course




	###################################
	# deciding the course of the lane #
	###################################

	def calc_lane_width(self, lane_matrix):
			
		left = lane_matrix[0,1]
		right = lane_matrix[0,3]
		lane_calc_width = right - left

		return lane_calc_width




	#####################
	# PUBLISH LANE INFO #
	#####################

	def lane_info_publischer(self, flag_lane_found, course):
			
		lane_info_msg = Int8MultiArray()
		lane_info_msg.data = [flag_lane_found, course]
		print(flag_lane_found)
		print(course)
		print(lane_info_msg.data)
		
		try:			
			self.lane_info.publish(lane_info_msg)
		except KeyError:
			pass
	
	######################################
	# Visualisation rectified FoI ZEDcam #
	######################################

	def make_rect_FoI_ZEDcam(self, lane_matrix, blur_aoi):

		# convert gray to color for visualisation (8uc1 to 8uc3)
		bird_lane = cv.cvtColor(blur_aoi, cv.COLOR_GRAY2RGB) #convert 8uc1 into 8uc3
			
		# calculate width for the lines and points
		thickness = np.int32(self.height_rect_aoi /500)
		k = 0
		scanrows = self.number_of_scanrows
		while k < scanrows: 
			# visualising the sections
			bird_lane = cv.line(bird_lane,(0,lane_matrix[k,0]),(self.width_rect_aoi,lane_matrix[k,0]),self.color_blue,thickness)
			k += 1
			
		# visualising a rectangle
		bird_lane = cv.rectangle(bird_lane,(self.corner_left_in_rect_aoi,self.height_rect_aoi),(self.corner_right_in_rect_aoi,0), self.color_green,thickness*2)
		
		# visualising the carcenter
		bird_lane = cv.line(bird_lane, (self.car_center_rect_aoi_left, self.height_rect_aoi),(self.car_center_rect_aoi_left, np.int32(self.height_rect_aoi/10*9)),(200,20,180),thickness*3) # center of car in rect aoi
	
		self.image_pub21.publish(self.bridge.cv2_to_imgmsg(bird_lane, "bgr8"))


	######################################
	# Visualisation Lane Detecton ZEDcam #
	######################################

	def visualisation_of_birdview(self, flag_dashed, lane_matrix, blur_aoi):
		
		if self.stopwatch_flag:
			start_visual_bird = time.time()

		# convert gray to color for visualisation (8uc1 to 8uc3)
		bird_lane = cv.cvtColor(blur_aoi, cv.COLOR_GRAY2RGB) #convert 8uc1 into 8uc3
		
		if self.stopwatch_flag:
			stop_col_to_gray = time.time()
			self.time_col_to_gray = stop_col_to_gray - start_visual_bird

			start_cv_drawings = time.time()

		# calculate width for the lines and points
		thickness = np.int32(self.height_rect_aoi /300)
		k = 0
		scanrows = self.number_of_scanrows
		while k < scanrows: 
			
			# visualising the windows
			bird_lane = cv.rectangle(bird_lane,(lane_matrix[k,4],lane_matrix[k,0]),(lane_matrix[k,5],lane_matrix[k,0]-5),self.color_blue,thickness) #left windows
			bird_lane = cv.rectangle(bird_lane,(lane_matrix[k,6],lane_matrix[k,0]),(lane_matrix[k,7],lane_matrix[k,0]-5),(220,220,10),thickness) # right windows

			if flag_dashed in (-1, 1):
				# visualising the windows
				bird_lane = cv.rectangle(bird_lane,(lane_matrix[k,13],lane_matrix[k,0]),(lane_matrix[k,14],lane_matrix[k,0]-5),self.color_gray,thickness) #left windows
				bird_lane = cv.rectangle(bird_lane,(lane_matrix[k,15],lane_matrix[k,0]),(lane_matrix[k,16],lane_matrix[k,0]-5),self.color_gray,thickness) # right windows

			k += 1
				
		# get the points of the lanes out of the matrix for the visualisation
		left_lane_pts 	= np.array(lane_matrix[:,[1,0]], np.int32)
		center_lane_pts = np.array(lane_matrix[:,[2,0]], np.int32)
		right_lane_pts 	= np.array(lane_matrix[:,[3,0]], np.int32)
			
		left_lane_pts =		left_lane_pts.reshape((-1,1,2))
		center_lane_pts = 	center_lane_pts.reshape((-1,1,2))
		right_lane_pts= 	right_lane_pts.reshape((-1,1,2))
			
		# visualising the lane lines
		bird_lane = cv.polylines(bird_lane,[left_lane_pts],False, self.color_green,thickness*2)
		bird_lane = cv.polylines(bird_lane,[center_lane_pts],False, self.color_red,thickness*2)
		bird_lane = cv.polylines(bird_lane,[right_lane_pts],False, self.color_green, thickness*2)

		if flag_dashed in (-1, 1):			
			# get the points of the lanes out of the matrix for the visualisation
			neighbor_left_lane_pts 	=	np.array(lane_matrix[:,[10,0]], np.int32)
			neighbor_center_lane_pts =	np.array(lane_matrix[:,[11,0]], np.int32)
			neighbor_right_lane_pts =	np.array(lane_matrix[:,[12,0]], np.int32)
				
			neighbor_left_lane_pts	= 	neighbor_left_lane_pts.reshape((-1,1,2))
			neighbor_center_lane_pts =	neighbor_center_lane_pts.reshape((-1,1,2))
			neighbor_right_lane_pts	=	neighbor_right_lane_pts.reshape((-1,1,2))
			
			# visualising the lane lines
			bird_lane = cv.polylines(bird_lane,[neighbor_left_lane_pts],False, self.color_gray,thickness)
			bird_lane = cv.polylines(bird_lane,[neighbor_center_lane_pts],False, self.color_gray,thickness*2)
			bird_lane = cv.polylines(bird_lane,[neighbor_right_lane_pts],False, self.color_gray, thickness)
			
		# visualising the carcenter
		bird_lane = cv.line(bird_lane, (self.car_center_rect_aoi_left, self.height_rect_aoi),(self.car_center_rect_aoi_left, np.int32(self.height_rect_aoi/10*9)),(200,20,180),thickness*3) # center of car in rect aoi

		# getting values from Matrix
		first_point_x = 		self.info_matrix[0,0]
		first_point_y = 		self.info_matrix[0,1]
		second_point_x = 		self.info_matrix[1,0]
		second_point_y = 		self.info_matrix[1,1]
		center_point_x = 		self.info_matrix[2,0]
		center_point_y = 		self.info_matrix[2,1]
		lateral_displacement =  self.info_matrix[0,2]
		radius =				self.info_matrix[1,2]
		course =				self.info_matrix[2,2]
		flag_lane_found = 		self.info_matrix[3,2]
		lane_calc_width =		self.info_matrix[4,2]
		
		# deciding the course text
		if course == 0:
			course_text = "Straight Lane"
		elif course < 0:
			course_text = "Left Curve"
		elif course > 0:
			course_text = "Right Curve"

		# deciding the text for lane
		if flag_lane_found == 1:
			lane_found_text = "Lane Detected"
		else:
			lane_found_text = "No Lane"

		# get values in Meter
		lateral_displacement_m = np.round(lateral_displacement*self.pixel2meter, 3)
		radius_m = np.abs(np.round(radius*self.pixel2meter, 3))
		lane_calc_width_m = np.round(lane_calc_width*self.pixel2meter, 3)
			
		# Draw part of circle
		bird_lane = cv.ellipse(bird_lane, (center_point_x,center_point_y), (np.abs(radius), np.abs(radius)), 0, 0, -180, (155,25,220), thickness) 
		ptls = np.array([[first_point_x,first_point_y],[center_point_x,center_point_y],[second_point_x,second_point_y]], np.int32)
		ptls = ptls.reshape((-1,1,2))
		bird_lane = cv.polylines(bird_lane,[ptls],False,(155,25,220), thickness)
	
		#text
		bird_lane = cv.putText(bird_lane, "Lane Status: " + lane_found_text,
						 				 np.int32((self.width_rect_aoi *0.02, self.height_rect_aoi *0.75)),
										 cv.FONT_HERSHEY_SIMPLEX, int(self.width_rect_aoi *0.0006), 
										 self.color_blue, int(self.width_rect_aoi *0.0008), cv.LINE_AA)

		bird_lane = cv.putText(bird_lane, "Lateral Displacement: " + str(lateral_displacement_m) +" m",
						 				 np.int32((self.width_rect_aoi *0.02, self.height_rect_aoi *0.80)),
										 cv.FONT_HERSHEY_SIMPLEX, int(self.width_rect_aoi *0.0006), 
										 self.color_blue, int(self.width_rect_aoi *0.0008), cv.LINE_AA)

		bird_lane = cv.putText(bird_lane, "Radius of Curve: " + str(radius_m) +" m",
						 				 np.int32((self.width_rect_aoi *0.02, self.height_rect_aoi *0.85)),
										 cv.FONT_HERSHEY_SIMPLEX, int(self.width_rect_aoi *0.0006), 
										 self.color_blue, int(self.width_rect_aoi *0.0008), cv.LINE_AA)
		
		bird_lane = cv.putText(bird_lane, course_text,
						 				 np.int32((self.width_rect_aoi *0.02, self.height_rect_aoi *0.90)),
										 cv.FONT_HERSHEY_SIMPLEX, int(self.width_rect_aoi *0.0006), 
										 self.color_blue, int(self.width_rect_aoi *0.0008), cv.LINE_AA)
		
		bird_lane = cv.putText(bird_lane, "Lane width: " + str(lane_calc_width_m) +" m",
						 				 np.int32((self.width_rect_aoi *0.02, self.height_rect_aoi *0.95)),
										 cv.FONT_HERSHEY_SIMPLEX, int(self.width_rect_aoi *0.0006), 
										 self.color_blue, int(self.width_rect_aoi *0.0008), cv.LINE_AA)
				
		if self.stopwatch_flag:
			stop_cv_drawings = time.time()
			self.time_cv_drawings = stop_cv_drawings - start_cv_drawings
		
		if self.stopwatch_flag:
			start_pub_bird_lane = time.time()

		# publishing the bird_lane
		self.image_pub22.publish(self.bridge.cv2_to_imgmsg(bird_lane, "bgr8"))
				
		if self.stopwatch_flag:
			stop_pub_bird_lane = time.time()
			self.time_pub_bird_lane = stop_pub_bird_lane - start_pub_bird_lane
			
			stop_visual_bird = time.time()
			self.time_visual_bird = stop_visual_bird - start_visual_bird

		
			
		

	###########################
	# DRAW THE DASHBOARD VIEW #
	###########################

	def dashboard_view(self, cv_image_orig, blur_aoi, lane_matrix):

		if self.stopwatch_flag:
			start_dashboard = time.time()

		# get lane_infos
		lateral_displacement =  self.info_matrix[0,2]
		radius =				self.info_matrix[1,2]
		course =				self.info_matrix[2,2]
		flag_lane_found = 		self.info_matrix[3,2]
		lane_calc_width =		self.info_matrix[4,2]
		flag_dashed	=			self.info_matrix[5,2]

		# get values in Meter
		lateral_displacement_m = 	np.round(lateral_displacement*self.pixel2meter, 3)
		radius_m =					np.round(radius*self.pixel2meter, 3)
		lane_calc_width_m =			np.round(lane_calc_width*self.pixel2meter, 3)

		# decition how to differ the dahs view 
		if flag_lane_found == 1:
			lateral_displacement_text = str(lateral_displacement_m) +" m"
			radius_text = str(np.abs(radius_m)) +" m"

			# deciding the course text
			if course == 0:
				course_text = "Straight Lane"
			elif course > 0:
				course_text = "Left Curve"
			elif course < 0:
				course_text = "Right Curve"

			# deciding dashed text
			if flag_dashed == 0:
				dashed_text = "No lane change possible"
			elif flag_dashed == -1:
				dashed_text = "Lane Change to Left Lane possible"
			else:
				dashed_text = "Lane Change to Right Lane possible"

			lane_width_text = str(lane_calc_width_m) +" m"
			color_flip = self.color_green
			text_flip = "GO"
			pos_flip = 0.7

		else:
			lateral_displacement_text = ""
			radius_text = ""
			course_text = ""
			dashed_text = ""
			lane_width_text = ""
			color_flip = self.color_red
			text_flip = "STOP"
			pos_flip = 0.6

		# calculate width for the lines and points
		thickness = np.int32(self.height_rect_aoi /500)

		# convert gray to color for visualisation (8uc1 to 8uc3)
		blur_aoi = cv.cvtColor(blur_aoi, cv.COLOR_GRAY2RGB) #convert 8uc1 into 8uc3
		overlay1 = np.zeros_like(blur_aoi)
		overlay2 = np.zeros_like(blur_aoi)
		
		# get the points of the lanes out of the matrix for the visualisation
		left_lane_pts 	= np.array(lane_matrix[:,[1,0]], np.int32)
		right_lane_pts 	= np.array(lane_matrix[:,[3,0]], np.int32)
		
		# make the matrix for the green carpet
		lane_area_left = left_lane_pts.copy()
		lane_area_right = right_lane_pts.copy()
		lane_area_left[:,0] += 20	
		lane_area_right[:,0] -= 20	
		lane_area_right_flipped = lane_area_right[::-1]
		lane_area = np.vstack((lane_area_left,lane_area_right_flipped))	
		lane_area = lane_area.reshape((-1,1,2))
				
		# draw lane lines and carpet
		overlay1 = cv.polylines(overlay1,[left_lane_pts],False, color_flip, thickness*4)
		overlay1 = cv.polylines(overlay1,[right_lane_pts],False, color_flip, thickness*4)
		overlay2 = cv.fillPoly(overlay2,[lane_area], color_flip)

		if flag_dashed in (-1, 1):
			# get the points of the lanes out of the matrix for the visualisation
			neighbor_left_lane_pts 	= np.array(lane_matrix[:,[10,0]], np.int32)
			neighbor_right_lane_pts = np.array(lane_matrix[:,[12,0]], np.int32)
			
			# make the matrix for the gray carpet
			neighbor_lane_area_left = neighbor_left_lane_pts.copy()
			neighbor_lane_area_right = neighbor_right_lane_pts.copy()
			neighbor_lane_area_left[:,0] += 30	
			neighbor_lane_area_right[:,0] -= 30	
			neighbor_lane_area_right_flipped = neighbor_lane_area_right[::-1]
			neighbor_lane_area = np.vstack((neighbor_lane_area_left, neighbor_lane_area_right_flipped))	
			neighbor_lane_area = neighbor_lane_area.reshape((-1,1,2))
					
			# draw carpet of neighbour lane
			overlay2 = cv.fillPoly(overlay2,[neighbor_lane_area], self.color_white)
		
		# combine lane lines and carpet
		overlay = cv.addWeighted(overlay1, 1, overlay2, 0.2, 0)

		# image transformation			image,			trans matrix			(width, height)
		overlay = cv.warpPerspective(overlay, self.inv_rect_trans_matrix, (self.x_dim_foi, self.y_dim_foi))#, flags=(cv.INTER_LINEAR))
		
		# Add a black border to the top of the image
		overlay = cv.copyMakeBorder(overlay, self.y_foi_horizon, 0, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
				
		# Frame of interest of the left zed imgage
		dash = cv_image_orig[:, self.x_dim_min:self.x_dim_foi] #cropded frame of picture for the lanedetection

		# combine dash view and the overlay
		dash = cv.addWeighted(dash, 1, overlay, 1, 0)

		# draw white field for info window
		field = np.zeros_like(dash)
		field = cv.rectangle(field, np.int32((self.x_dim_center *0.03, self.y_dim_max *0.05)),
									np.int32((self.x_dim_center *0.97, self.y_dim_max *0.45)),
									self.color_white, -1) 
		
		# combine dash and info window
		dash = cv.addWeighted(dash, 1, field, 0.5, 0)

		thickness = np.float32(self.x_dim_center /1100)
		#text
		dash = cv.putText(dash, "Lateral Displacement: " + lateral_displacement_text,
						 				 np.int32((self.x_dim_center *0.04, self.y_dim_max *0.10)),
										 cv.FONT_HERSHEY_SIMPLEX, thickness, 
										 self.color_blue, int(1), cv.LINE_AA)
		
		dash = cv.putText(dash, "Radius of the Curve: " + radius_text,
						 				 np.int32((self.x_dim_center *0.04, self.y_dim_max *0.18)),
										 cv.FONT_HERSHEY_SIMPLEX, thickness, 
										 self.color_blue, int(1), cv.LINE_AA)
		
		dash = cv.putText(dash, "Course of the Lane: " + str(course_text),
						 				 np.int32((self.x_dim_center *0.04, self.y_dim_max *0.26)),
										 cv.FONT_HERSHEY_SIMPLEX, thickness, 
										 self.color_blue, int(self.x_dim_center *0.0008), cv.LINE_AA)
		
		dash = cv.putText(dash, str(dashed_text),
						 				 np.int32((self.x_dim_center *0.04, self.y_dim_max *0.34)),
										 cv.FONT_HERSHEY_SIMPLEX, thickness, 
										 self.color_blue, int(self.x_dim_center *0.0008), cv.LINE_AA)
		
		dash = cv.putText(dash, "Lane Width: " + lane_width_text,
						 				 np.int32((self.x_dim_center *0.04, self.y_dim_max *0.42)),
										 cv.FONT_HERSHEY_SIMPLEX, thickness, 
										 self.color_blue, int(self.x_dim_center *0.0008), cv.LINE_AA)
				
		dash = cv.putText(dash, text_flip,
										np.int32((self.x_dim_center *pos_flip, self.y_dim_max *0.26)),
										cv.FONT_HERSHEY_SIMPLEX, thickness*4, 
										color_flip, int(self.x_dim_center *0.01), cv.LINE_AA)
	

		# publish dashboard
		self.image_pub2.publish(self.bridge.cv2_to_imgmsg(dash, "bgr8"))
		
		# time
		if self.stopwatch_flag:
			stop_dashboard = time.time()
			self.time_dashboard = stop_dashboard - start_dashboard



	###########################
	# Measuring the Trapezoid #
	###########################

	#here the picture gets converted and the pixel whrere the left lane an dthe right lane begin from the outside are calculated
	def measure_trapezoid(self, cv_image_orig):
		
		# Frame of interest of the left zed imgage
		cv_image_left_foi = cv_image_orig[self.y_foi_horizon:self.y_dim_max, self.x_dim_min:self.x_dim_center] #cropded frame of picture for the lanedetection
				
		# turn color in to black and white
		image_red_hls = cv.cvtColor(cv_image_left_foi, cv.COLOR_BGR2HLS) #gives every bixeltransformiert das RGB Bild in Grauwerte von 0 bis 188 0:hue, farbe 1:grayValus, 2:saturation sättigung 
	
		_ , image_red_threshold = cv.threshold(image_red_hls[:,:,1], 100, 255, cv.THRESH_BINARY) #original(100,255) #dimensions [y_pixel,x_pixel,informations], sets a threshold on the grayvalues to make the picture black and white
		
		blur_image = cv.GaussianBlur(image_red_threshold, (3,3), 0) #blurring the black and white picture
		
		lane_markings_bw = blur_image
				
		#sums up all the values of each y clumn of the 640x5 picture. So now there is a value vor every of the 640 x pixelk
		self.histogram1 = np.sum(lane_markings_bw[self.y_dim_foi -5:self.y_dim_foi, :], axis=0) #the y row foresight starts at the 50th pixel and ends at the 55, 
		self.histogram2 = np.sum(lane_markings_bw[0:5, :], axis=0) #the y row foresight starts at the 50th pixel and ends at the 55, 

		#puts all the x coordinates where the sum is >1000 in an array		
		indices1 = np.asarray(np.argwhere(self.histogram1 > 1000)) #255*5=1275 is the max value and 1000 is the threshold
		indices2 = np.asarray(np.argwhere(self.histogram2 > 700)) #255*5=1275 is the max value and 1000 is the threshold
				
		# presetting the valid image flag
		valid_image = True

		# Scanning the Bottom from the outside to the inside
		if len(indices1) != 0:
			# get the indices of the fist recognised lane from the outside to the inside
			left_lane = np.min(indices1)#np.argmax(self.histogram1[:self.histogram1.shape[0]/2])
			right_lane = np.max(indices1)#np.argmax(self.histogram1[self.histogram1.shape[0]/2:])+self.histogram1.shape[0]/2
											
		else:
			valid_image = False
			error_msg = "No lines detected at the bottom of the image"
			self.get_logger().info(error_msg) 
		
		# Scanning the horizon from the middle to the outside
		if len(indices2) != 0:
			# get the first recogniced lanes from the inside to the outside
			# it starts searching on the car_center
			minusplusmiddle2 = ((indices2/self.car_center_img_left)-1)

			# np.where puts all the incides of the array wich values meet the condition in to one array.
			# all the indices wich don't are put into an secont array with zeros.
			# so only the first array is important -> np.where()[0]			
			if len(np.where(minusplusmiddle2 < 0)[0]) != 0:
				index_left2 = np.max(np.where(minusplusmiddle2 < 0)[0])
				left_lane2 = int(indices2[index_left2])
			else:
				valid_image = False
				error_msg = "No left line detected at the horizon of the image"
				self.get_logger().info(error_msg) 
				
			if len(np.where(minusplusmiddle2 > 0)[0]) != 0:
				index_right2 = np.min(np.where(minusplusmiddle2 > 0)[0]) 
				right_lane2 = int(indices2[index_right2])
			else:
				valid_image = False
				error_msg = "No right line detected at the horizon of the image"
				self.get_logger().info(error_msg) 
		else:
			valid_image = False
			error_msg = "No lines detected at the horizon of the image"
			self.lane_is_detected = False
							
		#Visualisation
		thickness = np.int32(self.y_dim_max /200)
		
		# Left image of the zed
		cv_image_left = cv_image_orig[:, self.x_dim_min:self.x_dim_center, :] #croping the image of the zed for visualisation	

		# blue lines for the horizon and the vertical image of the lane
		lane_markings = cv.line(cv_image_left, (self.img_left_center, self.y_dim_min), 	(self.img_left_center, self.y_dim_max), 	(255,200,50), thickness*2) #left image center blue
		lane_markings = cv.line(lane_markings, (self.x_dim_min, self.y_foi_horizon),	(self.x_dim_center, self.y_foi_horizon),	(255,200,50), thickness*2) #horizon line

		# visualisation
		if valid_image:
			# left marker
			lane_markings = cv.drawMarker(lane_markings, (left_lane2, self.y_foi_horizon-thickness*10), self.color_green, cv.MARKER_TRIANGLE_DOWN, thickness*15, thickness*2)
			lane_markings = cv.putText(lane_markings, str(left_lane2) +"pixel",
											np.int32((left_lane2-thickness*60, self.y_foi_horizon-thickness*40)),
											cv.FONT_HERSHEY_SIMPLEX, thickness, 
											self.color_green, int(1), cv.LINE_AA)
			
			# right marker
			lane_markings = cv.drawMarker(lane_markings, (right_lane2, self.y_foi_horizon-thickness*10), self.color_green, cv.MARKER_TRIANGLE_DOWN, thickness*15, thickness*2)
			lane_markings = cv.putText(lane_markings, str(right_lane2) +"pixel",
											np.int32((right_lane2-thickness*60, self.y_foi_horizon-thickness*40)),
											cv.FONT_HERSHEY_SIMPLEX, thickness, 
											self.color_green, int(1), cv.LINE_AA)

			# construct the matrix of the trapezoid for the cv.fillPoly function
			trapezoid = np.zeros_like(lane_markings)
			trapezoid_pts 	= np.array(([left_lane, self.y_dim_max],
									[left_lane2, self.y_foi_horizon],
									[right_lane2, self.y_foi_horizon],
									[right_lane, self.y_dim_max]), np.int32)
			trapezoid_pts = trapezoid_pts.reshape((-1,1,2))
			
			# draw the trapezoid area
			trapezoid = cv.fillPoly(trapezoid,[trapezoid_pts], self.color_green)
			
			# combine lanemarkings and trapezoid
			lane_markings = cv.addWeighted(lane_markings, 1, trapezoid, 0.2, 0)

			# draw the trapezoid frame
			lane_markings = cv.polylines(lane_markings,[trapezoid_pts],True, self.color_green,thickness*2)
			
			self.get_logger().info("upper left:" + str(left_lane2) + "and right: " + str(right_lane2)) 

		else:
			lane_markings = cv.putText(lane_markings, error_msg,
											np.int32((self.y_dim_foi*0.1, self.y_foi_horizon-thickness*40)),
											cv.FONT_HERSHEY_SIMPLEX, thickness*0.8, 
											self.color_red, int(1), cv.LINE_AA)

		self.image_pub20.publish(self.bridge.cv2_to_imgmsg(lane_markings, "bgr8"))
		




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
