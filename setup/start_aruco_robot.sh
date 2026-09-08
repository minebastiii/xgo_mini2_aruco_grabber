#!/bin/bash
# /home/pi/start_aruco_robot.sh
# Wrapper für den aruco-robot systemd Service.
# Angepasst für ROS 2 Jazzy

set -e

# ROS 2 base sourcing
source /opt/ros/jazzy/setup.bash
source /home/pi/venv/bin/activate

# Workspace sourcing (nur wenn vorhanden)
WORKSPACE="$HOME/workspace/install/setup.bash"
if [ -f "$WORKSPACE" ]; then
    source "$WORKSPACE"
fi

export ROS_DOMAIN_ID=34

exec ros2 launch aruco_robot robot_new.launch.py
