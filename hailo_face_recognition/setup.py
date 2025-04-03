from setuptools import setup
import os
from glob import glob

package_name = "hailo_face_recognition"

setup(
    name=package_name,
    version="1.0.3",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join(package_name, "launch", "*")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join(package_name, "config", "*")),
        ),
        (
            os.path.join("share", package_name, "resources"),
            glob(os.path.join(package_name, "resources", "*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Stefanos Kyrikakis",
    maintainer_email="kirikakis@gmail.com",
    description="Hailo Face Regognition ROS2 Package",
    license="Apache 2.0 License",
    tests_require=["pytest", "pydocstyle"],
    entry_points={
        "console_scripts": [
            "face_recognition_node = hailo_face_recognition.face_recognition_node:main",
        ],
    },
)
