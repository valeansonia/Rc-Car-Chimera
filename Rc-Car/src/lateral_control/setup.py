from setuptools import setup
import os
from glob import glob

package_name = 'lateral_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@ubuntu.com',
    description='Detects lanes on surface',
    license='Detects lanes on surface',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'in_lane_positioning = lateral_control.in_lane_positioning:main',
            'lane_information = lateral_control.lane_information:main',
            'steering_control = lateral_control.steering_control:main',
            'lane2 = lateral_control.lane2:main',   
            'visual = lateral_control.visual:main',
            'traffic_sign_classifier = lateral_control.traffic_sign_classifier:main',
            'traffic_light_detector = lateral_control.traffic_light_detector:main',
            'traffic_light_drive_policy = lateral_control.traffic_light_drive_policy:main',
            'traffic_light_yolo_detector = lateral_control.traffic_light_yolo_detector:main',
            'traffic_light_zed_test = lateral_control.traffic_light_zed_test:main',
            'ack_to_pca = lateral_control.ack_to_pca:main',
            'lidar_detection = lateral_control.lidar_detection:main',
            'autonomous_video_version = lateral_control.autonomous_video_version:main',
        ],
    },
)
