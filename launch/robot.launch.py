"""robot.launch.py — launch on the robot (Pi)"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("width",         default_value="1920"),
        DeclareLaunchArgument("height",        default_value="1080"),
        DeclareLaunchArgument("fps",           default_value="15"),
        DeclareLaunchArgument("jpeg_quality",  default_value="60"),

        # ── Camera node ──────────────────────────────────────────────
        Node(
            package="aruco_robot",
            executable="camera_node",
            name="camera_node",
            output="screen",
            parameters=[{
                "width":        LaunchConfiguration("width"),
                "height":       LaunchConfiguration("height"),
                "fps":          LaunchConfiguration("fps"),
                "jpeg_quality": LaunchConfiguration("jpeg_quality"),
                "publish_raw":        False,
                "publish_compressed": True,
            }],
        ),

        # ── FSM / control node ───────────────────────────────────────
        Node(
            package="aruco_robot",
            executable="fsm_node",
            name="fsm_node",
            output="screen",
            parameters=[{
                "center_threshold":  40.0,
                "approach_area":   6000.0,
                "search_turn_speed": 0.4,
                "approach_speed":    0.3,
                "image_width":    LaunchConfiguration("width"),
                "image_height":   LaunchConfiguration("height"),
            }],
        ),
    ])
