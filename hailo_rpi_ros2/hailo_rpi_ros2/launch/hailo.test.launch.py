# Copyright 2025 Stefanos Kyrikakis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# !/usr/bin/env python3


import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    hailo_params = os.path.join(
        get_package_share_directory("hailo_rpi_ros2"), "config", "hailo_params.yaml"
    )

    hailo_detection_node = Node(
        package="hailo_rpi_ros2",
        executable="face_recognition_node",
        name="face_recognition",
        output="screen",
        parameters=[
            hailo_params,
            {
                "input": (
                    "/workspaces/src/hailo_rpi_ros2/hailo_rpi_ros2/hailo_rpi_ros2/"
                    "resources/face_recognition.mp4"
                )
            },
            {"local_gallery_file": ("face_recognition_local_gallery.json")},
            {"similarity_threshhold": 0.40}
        ],
    )

    return LaunchDescription([hailo_detection_node])
