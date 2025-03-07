from setuptools import setup
import subprocess

package_name = 'hailo_rpi_ros2'

# Execute install.sh
try:
    subprocess.run(['bash', 'install.sh'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running install.sh: {e}")
    raise RuntimeError("install.sh failed. Installation aborted.") #Raise the exception.

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/srv', ['srv/AddPerson.srv'])
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
            'hailo_rpi_ros2 = hailo_rpi_ros2_pkg.ros2_node:main',
        ],
    },
)

# Post-install script
try:
    subprocess.run(['bash', 'post_install.sh'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running post_install.sh: {e}")
    # Optionally, raise an exception to stop the installation