"""robot.launch.py — launch on the robot (Pi)"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            # ── Camera-Argumente ─────────────────────────────────────────
            DeclareLaunchArgument("width", default_value="1920"),
            DeclareLaunchArgument("height", default_value="1080"),
            DeclareLaunchArgument("fps", default_value="15"),
            DeclareLaunchArgument("jpeg_quality", default_value="60"),
            # ArUco / Pose
            DeclareLaunchArgument("marker_length_m", default_value="0.10"),
            DeclareLaunchArgument("aruco_dict", default_value="DICT_4X4_50"),
            DeclareLaunchArgument("target_marker_id", default_value="0"),
            DeclareLaunchArgument("trailer_marker_id", default_value="1"),
            # Kamera-Kalibrierung — Platzhalter, unbedingt echte Werte eintragen!
            DeclareLaunchArgument("camera_fx", default_value="1400.0"),
            DeclareLaunchArgument("camera_fy", default_value="1400.0"),
            DeclareLaunchArgument("camera_cx", default_value="960.0"),
            DeclareLaunchArgument("camera_cy", default_value="540.0"),
            # ── FSM-Argumente ────────────────────────────────────────────
            DeclareLaunchArgument("lateral_tolerance_m", default_value="0.03"),
            DeclareLaunchArgument("lateral_gain", default_value="8.0"),
            DeclareLaunchArgument("lateral_sign", default_value="1.0"),
            # ── Camera node ──────────────────────────────────────────────
            Node(
                package="aruco_robot",
                executable="camera_pose_node",
                name="camera_pose_node",
                output="screen",
                parameters=[
                    {
                        "width": LaunchConfiguration("width"),
                        "height": LaunchConfiguration("height"),
                        "fps": LaunchConfiguration("fps"),
                        "jpeg_quality": LaunchConfiguration("jpeg_quality"),
                        "publish_raw": False,
                        "marker_length_m": LaunchConfiguration("marker_length_m"),
                        "aruco_dict": LaunchConfiguration("aruco_dict"),
                        "target_marker_id": LaunchConfiguration("target_marker_id"),
                        "trailer_marker_id": LaunchConfiguration("trailer_marker_id"),
                        "camera_fx": LaunchConfiguration("camera_fx"),
                        "camera_fy": LaunchConfiguration("camera_fy"),
                        "camera_cx": LaunchConfiguration("camera_cx"),
                        "camera_cy": LaunchConfiguration("camera_cy"),
                        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                ],
            ),
            # ── FSM / control node ───────────────────────────────────────
            Node(
                package="aruco_robot",
                executable="fsm_pose_node",
                name="fsm_pose_node",
                output="screen",
                parameters=[
                    {
                        # "target_marker_area":    22500.0,
                        # "container_marker_area": 60000.0,
                        # "grasp_distance_m":      0.15,
                        # "deploy_distance_m":     0.15,
                        "lateral_tolerance_m": LaunchConfiguration(
                            "lateral_tolerance_m"
                        ),
                        "lateral_gain": LaunchConfiguration("lateral_gain"),
                        "lateral_sign": LaunchConfiguration("lateral_sign"),
                    }
                ],
            ),
        ]
    )
