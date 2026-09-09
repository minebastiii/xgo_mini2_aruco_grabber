#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
source /home/pi/venv/bin/activate

WORKSPACE="$HOME/workspace/install/setup.bash"
if [ -f "$WORKSPACE" ]; then
    source "$WORKSPACE"
fi

export ROS_DOMAIN_ID=34

exec ros2 launch aruco_robot robot.launch.py
