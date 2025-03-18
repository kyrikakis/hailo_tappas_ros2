#!/bin/bash
source /opt/ros/$ROS_DISTRO/setup.bash && \
source /workspaces/install/setup.bash
ros2 launch hailo_face_recognition hailo.launch.py