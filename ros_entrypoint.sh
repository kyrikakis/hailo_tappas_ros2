#!/bin/bash
set -e

id -u ros &>/dev/null || adduser --quiet --disabled-password --gecos '' --uid ${UID:=1200} --uid ${GID:=1200} ros

getent group i2c || groupadd -g 998 i2c
usermod -aG i2c ros

# setup ros environment
export ROS_DISTRO=jazzy
export ROS_DOMAIN_ID=20
source "/opt/ros/$ROS_DISTRO/setup.bash"

/usr/bin/supervisord
exec "$@"
