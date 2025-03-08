from setuptools import setup
import os

package_name = 'hailo_rpi_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stefanos Kyrikakis',
    maintainer_email='kirikakis@gmail.com',
    description='ROS 2 package for Hailo RPi',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_node = hailo_rpi_ros2.ros2_node:main',
        ],
    },
)