FROM debian:bookworm

SHELL ["/bin/bash", "-c"]

ENV ROS_DISTRO jazzy
ENV LANG en_US.UTF-8

# Install generic requirements
RUN apt-get update && \
    apt-get install -y software-properties-common wget

# Need to create a sources.list file for apt-add-repository to work correctly:
# https://groups.google.com/g/linux.debian.bugs.dist/c/6gM_eBs4LgE
RUN echo "# See sources.lists.d directory" > /etc/apt/sources.list

# Add Raspberry Pi repository, as this is where we will get the Hailo deb packages
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E && \
    apt-add-repository -y -S deb http://archive.raspberrypi.com/debian/ bookworm main

# Dependencies for hailo-tappas-core
RUN apt-get install -y python3 ffmpeg x11-utils python3-dev python3-pip \
    python3-setuptools gcc-12 g++-12 python-gi-dev pkg-config libcairo2-dev \
    libgirepository1.0-dev libgstreamer1.0-dev cmake \
    libgstreamer-plugins-base1.0-dev libzmq3-dev rsync git \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-libcamera libopencv-dev \
    python3-opencv

# Dependencies for rpicam-apps-hailo-postprocess
RUN apt-get install -y rpicam-apps hailo-tappas-core-3.28.2
# Excludes hailort as it fails to install during build stage

# Dependencies for hailo-rpi5-examples
RUN apt-get install -y python3-venv meson

# Download Raspberry Pi examples
RUN git clone --depth 1 https://github.com/raspberrypi/rpicam-apps.git

# Download Hailo examples
RUN git clone https://github.com/hailo-ai/hailo-rpi5-examples.git && \
    cd hailo-rpi5-examples && ./download_resources.sh

RUN git clone https://github.com/kyrikakis/tappas.git

RUN wget https://s3.ap-northeast-1.wasabisys.com/download-raw/dpkg/ros2-desktop/debian/bookworm/ros-jazzy-desktop-0.3.2_20240525_arm64.deb && \
    apt install -y ./ros-jazzy-desktop-0.3.2_20240525_arm64.deb && \
    pip install --break-system-packages vcstool psutil colcon-common-extensions

RUN echo "export ROS_DOMAIN_ID=20" >> ~/.bashrc
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc

COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x  /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

RUN mkdir -p /workspaces/hailo-rpi-ros2/
COPY . /workspaces/hailo-rpi-ros2/

USER $USERNAME
# terminal colors with xterm
ENV TERM xterm
WORKDIR /workspaces/hailo-rpi-ros2
CMD [ "sleep", "infinity" ]
