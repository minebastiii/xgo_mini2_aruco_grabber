from setuptools import setup
import os
from glob import glob

package_name = "aruco_robot"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.py")),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "camera_node = aruco_robot.camera_node:main",
            "fsm_node    = aruco_robot.fsm_node:main",
            "camera_pose_node = aruco_robot.camera_pose_node:main",
            "fsm_pose_node = aruco_robot.fsm_pose_node:main",
            "camera_node_new = aruco_robot.camera_node_new:main",
            "fsm_node_new = aruco_robot.fsm_node_new:main",
            "grasp_test = aruco_robot.grasp_test:main",
        ],
    },
)
