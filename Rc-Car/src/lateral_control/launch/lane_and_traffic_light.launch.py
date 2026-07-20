import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("lateral_control")
    lane_launch = os.path.join(
        package_share,
        "launch",
        "lateral_control.launch.py",
    )

    model_path = LaunchConfiguration("model_path")
    show_window = LaunchConfiguration("show_window")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value="src/model/bestSem.pt",
                description="Path to the traffic-light YOLO .pt model",
            ),
            DeclareLaunchArgument(
                "show_window",
                default_value="false",
                description="Show the OpenCV traffic-light debug window",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(lane_launch)
            ),
            Node(
                package="lateral_control",
                executable="traffic_light_camera_node",
                name="traffic_light_camera_node",
                output="screen",
                parameters=[
                    {
                        "image_topic": "/ZEDcam/image_raw",
                        "model_path": model_path,
                        "show_window": ParameterValue(
                            show_window, value_type=bool
                        ),
                        "publish_debug_image": True,
                        "use_left_zed_image": True,
                    }
                ],
            ),
        ]
    )
