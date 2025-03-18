from setuptools import setup

package_name = "hailo_common"

setup(
    name=package_name,
    version="1.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Stefanos Kyrikakis",
    maintainer_email="kirikakis@gmail.com",
    description="ROS2 package for Hailo common classes",
    license="Apache 2.0 License",
    tests_require=["pytest", "pydocstyle"],
)
