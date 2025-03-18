# Hailo ROS2
This project bundles ROS2 and Hailo tappas together in a Debian docker container to maximise portability and scalability keeping you host clean from dependencies.

## Supported versions

* OS: Debian bookworm
* ROS: Jazzy
* HailoRT Drivers: 4.20.0
* hailo-tappas-core: 3.31.0

## Installation

### Install driver on the host

For the SDK to work with the AI accelerator hardware, a matching version of the driver needs to be used.
Even a minor version difference will prevent the SDK from detecting the hardware.

Install requirements on host:

```
sudo apt-get install linux-headers-$(uname -r)
```

We get the exact version of the driver's source code from Github:

```
git clone --depth 1 --branch v4.20.0 https://github.com/hailo-ai/hailort-drivers.git
```

Then build it and install it on the host system:

```
cd linux/pcie
make all
sudo make install
sudo modprobe hailo_pci
cd ../..
./download_firmware.sh
sudo mkdir -p /lib/firmware/hailo
sudo mv hailo8_fw.4.17.0.bin /lib/firmware/hailo/hailo8_fw.bin
sudo cp ./linux/pcie/51-hailo-udev.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

It's better to restart after installing this driver.
After a reboot you can look at the kernel buffer to see if the device is detected and the driver loaded.

```
$ sudo dmesg | grep hailo
...
[    4.379687] hailo: Init module. driver version 4.20.0
...
[    4.545602] hailo 0000:01:00.0: Firmware was loaded successfully
[    4.572371] hailo 0000:01:00.0: Probing: Added board 1e60-2864, /dev/hailo0
```
## Build the container and open a shell
```
docker compose build
docker compose up -d hailo-rpi-service
```
Open a shell inside the container

```
docker compose exec hailo-rpi-service /bin/bash
```
OR just open it with VSCode with the Dev Containers extension

# Use Cases

## Face recognition

A Face regognition ROS2 node based on tappas pipelines. Running face detection, face recognition and Yolo object detection in parallel. Publishing the detection data and video to ROS2 topics and exposing services for saving embeddings into a local gallery file. 

The application is getting the most out of the Hailo 8 while keeping the CPU consuption on the host in relative low levels. The sweet spot for running smoothly is at 15 fps, enough for most of the use cases. There is an parameter allowing you to skip the Yolo inference and then it can easily go up to 30 fps.

![Face Recognition and Yolo](face_recognition.gif)

## Test
Running the until tests:
```
cd /workspaces && \
colcon test --event-handlers console_direct+
```
The ros node will run on the container startup, stop it:
```
supervisorctl stop hailo
```
Running the end to end test:
```
ros2 launch hailo_face_recognition hailo.test.launch.py
```
## Run
The ros node will run on the container startup, if you want to stop it run:
```
supervisorctl stop hailo
```
If you want to run it directly:
```
ros2 launch hailo_face_recognition hailo.launch.py
```

# API

## Published topics

| Topic  | Type | Description |
|-----|----|----|
| /hailo_face_recognition/image_bbx/compressed | [`sensor_msgs.msg.Image`](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/Image.msg) | image frame with bounding boxes |
| /hailo_face_recognition/detections | [`vision_msgs.msg.Detection2DArray`](https://github.com/ros-perception/vision_msgs/blob/4.1.1/vision_msgs/msg/Detection2DArray.msg) | A list of 2D detections |

## Exposed Services

| Name  | Type | Description |
|-----|----|---------|
| /hailo_face_recognition/save_face  | [`hailo_msgs/srv/SaveFace`](hailo_msgs/srv/SaveFace.srv) | Adding or appending embedding from current detection to face |
| /hailo_face_recognition/delete_face  | [`hailo_msgs/srv/DeleteFace`](hailo_msgs/srv/DeleteFace.srv) | Deleting all face embeddings from face |

## Considerations

### APIs

Hailo tappas is exposing the following main identifiers: tracking_id, global_id, class_id and classification.

**tracking_id:** Is an identifier assigned by the tracking algorithm. It aims to maintain a consistent ID for the same object across consecutive video frames.

**global_id:** Is an identifier that is consistent between different runtimes of the application. Applies based on the embeddings distance only to unique detections, like faces or persons

**class_id:** Represents the category or class of the detected object. It indicates what type of object the model has identified (e.g., "person," "car," "face")

**classification:** Is a more general term that encompasses various types of classification results, It can include classification data like face recognition results, where the classification might be a person's name or attribute Classifications: (e.g., gender, age, emotion)

**hailo_bbox:** Exposes 4 float arguments representing the bounding box. The first 2 arguments are the x and y of the minimum point (Top Left corner). The other 2 arguments are the width and height of the box respectivly

**confidence:** The confidence of the detection from scale 0.0 to 1.0

Hailo <> ROS2 [`vision_msgs.msg.Detection2D`](https://github.com/ros-perception/vision_msgs/blob/4.1.1/vision_msgs/msg/Detection2D.msg) mappings:

| ROS2 Detection2D | Hailo Detection |
| ---------------  | --------------- |
| Detection2D.id   | tracing_id |
| Detection2D.results[0].hypothesis.class_id | class_id + ": " + classification |
| Detection2D.bbox | hailo_bbox |

As you can see above ROS2 Detection2D have provisions for only the **Detection2D.id** and the **ObjectHypothesis.class_id** as such in this solution when a face is presented and regognised the **classification** is concatenated in the **ObjectHypothesis.class_id**. The **global_id** is not getting mapped anywhere.

### Performance

In the case of the host is engaging only to facilitate the models inference and running the GStreamer pipeline the CPU consumption is really low at around **20%** per core on RPi5.

When using the embeddings_gallery like in the face recognition use case the complexity is **O(N*M)** where **N** the total number of embeddings and **M** the number of face detections in a single frame. In short a RPi5 can handle around **200** embeddings with **5** faces on the same frame. This could be further improved by comparing the embeddings in a multithreading fashion and/or porting this code to C++ or Rust.

### ROS Domain ID

The ROS domain ID is set to #20. Feel free to change that to you needs.

# Acknowledgements
* https://github.com/hailo-ai/hailo-rpi5-examples
* https://github.com/canonical/pi-ai-kit-ubuntu
* https://github.com/Ar-Ray-code/rpi-bullseye-ros2