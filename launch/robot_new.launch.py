"""robot.launch.py — launch on the robot (Pi)

Runs camera_node (capture + ArUco detection + pose estimation, all
local now) and fsm_node. detector_node.py is no longer launched
anywhere — it has been merged into camera_node.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # ── Camera capture ───────────────────────────────────────────
        DeclareLaunchArgument("width",         default_value="1920"),
        DeclareLaunchArgument("height",        default_value="1080"),
        DeclareLaunchArgument("fps",           default_value="15"),
        DeclareLaunchArgument("jpeg_quality",  default_value="60"),
        # ── ArUco / detector tuning (formerly detector_node args) ──────
        DeclareLaunchArgument("config_path",   default_value="aruco_config.yaml"),
        DeclareLaunchArgument("cam_fx",        default_value="1400.0"),
        DeclareLaunchArgument("cam_fy",        default_value="1400.0"),
        DeclareLaunchArgument("cam_cx",        default_value="960.0"),
        DeclareLaunchArgument("cam_cy",        default_value="540.0"),
        DeclareLaunchArgument("target_marker_size_m",  default_value="0.019"),
        DeclareLaunchArgument("trailer_marker_size_m", default_value="0.10"),

        # ── Camera node (camera capture + ArUco detection, merged) ────
        Node(
            package="aruco_robot",
            executable="camera_node_new",
            name="camera_node",
            output="screen",
            parameters=[{
                "width":        LaunchConfiguration("width"),
                "height":       LaunchConfiguration("height"),
                "fps":          LaunchConfiguration("fps"),
                "jpeg_quality": LaunchConfiguration("jpeg_quality"),
                "publish_raw":        False,
                "config_path":  LaunchConfiguration("config_path"),
                "cam_fx": LaunchConfiguration("cam_fx"),
                "cam_fy": LaunchConfiguration("cam_fy"),
                "cam_cx": LaunchConfiguration("cam_cx"),
                "cam_cy": LaunchConfiguration("cam_cy"),
                "target_marker_size_m":  LaunchConfiguration("target_marker_size_m"),
                "trailer_marker_size_m": LaunchConfiguration("trailer_marker_size_m"),
            }],
        ),

        # ── FSM / control node — unchanged, same topic contract ────────
        Node(
            package="aruco_robot",
            executable="fsm_node_new",
            name="fsm_node",
            output="screen",
            parameters=[{
                "image_width":  LaunchConfiguration("width"),
                "image_height": LaunchConfiguration("height"),
            }],
        ),
    ])
