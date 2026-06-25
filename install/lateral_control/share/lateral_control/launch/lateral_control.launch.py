import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
	ld = LaunchDescription()

	package_share_directory = get_package_share_directory('lateral_control')
	
	HDwebCam_config = os.path.join(package_share_directory, 'config', 'camera_config_HDwebCam.yaml')
	ZED_367p_config = os.path.join(package_share_directory, 'config', 'camera_config_ZED_376p.yaml')
	ZED_1242p_config = os.path.join(package_share_directory, 'config', 'camera_config_ZED_1242p.yaml')
	lanedtct_config = os.path.join(package_share_directory, 'config', 'lane_detection.yaml')
	steering_config = os.path.join(package_share_directory, 'config', 'steering_control.yaml')
	
	HDwebCam = Node(
		package='usb_cam',
		executable='usb_cam_node_exe',
		name='HDwebCam',
		parameters=[HDwebCam_config],
		remappings=[('/image_raw', '/HDwebCam/image_raw')]
	)

	ZEDcam = Node(
		package='usb_cam',
		executable='usb_cam_node_exe',
		name='ZEDcam',
		parameters=[ZED_367p_config],
		remappings=[('/image_raw', '/ZEDcam/image_raw')]
	)

	in_lane_positioning = Node(
		package='lateral_control',
		executable='in_lane_positioning',
		name='in_lane_positioning',
		parameters=[lanedtct_config],
	)

	lane_information = Node(
		package='lateral_control',
		executable='lane_information',
		name='lane_information',
		parameters=[lanedtct_config],
		output='screen'
	)

	steering_control = Node(
		package='lateral_control',
		executable='steering_control',
		name='steering_control',
		parameters=[steering_config]
	)	
	
	# finalize
	ld.add_action(HDwebCam)
	ld.add_action(ZEDcam)
	ld.add_action(in_lane_positioning)
	ld.add_action(lane_information)
	ld.add_action(steering_control)
	return ld
	
	
	

