cmake_minimum_required(VERSION 3.8)
project(hailo_rpi_ros2_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SaveFace.srv"
  "srv/DeleteFace.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()