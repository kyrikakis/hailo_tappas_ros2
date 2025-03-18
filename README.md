# Hailo tappas ROS2
[![Version](https://img.shields.io/badge/version-1.0.1-green.svg)](https://github.com/kyrikakis/hailo_tappas_ros2/releases/tag/v1.0.1)

This project streamlines the development and deployment of Hailo tappas applications within ROS 2, by providing a pre-configured, containerized environment. The project is also fully configured with Dev Containers using VS Code, enables rapid development of use cases, eliminating environment setup concerns, and delivers production-ready deployments with container auto-restart and supervisor support.

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
cd hailort-drivers/linux/pcie
make all
sudo make install
sudo modprobe hailo_pci
cd ../..
./download_firmware.sh
sudo mkdir -p /lib/firmware/hailo
sudo mv hailo8_fw.4.20.0.bin /lib/firmware/hailo/hailo8_fw.bin
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
docker compose up -d hailo-tappas-service
```
Open a shell inside the container

```
docker compose exec hailo-tappas-service /bin/bash
```
OR

This project is fully configured for Dev Containers, just open it in VSCode using the Dev Containers extension skipping the above steps.

# Use Cases

## Face recognition

A Face regognition ROS2 node based on this [example](https://github.com/hailo-ai/tappas/blob/v3.31.0/apps/h8/gstreamer/general/face_recognition/README.rst) tappas application. Running face detection, face recognition and object detection in parallel. Publishing the detection data and video to ROS2 topics and exposing services for saving embeddings into a local gallery file. 

The application is getting the most out of **Hailo 8** while keeping the CPU consuption on the host in relative low levels. The sweet spot for running smoothly is at 15 fps on HD video resolution, enough for most of the use cases. There is a parameter allowing you to skip the object detection inference and it can easily go up to 30 fps.

![Face Recognition and Yolo](hailo_face_recognition/hailo_face_recognition/resources/face_recognition.gif)

## Models

| Task | Model | HW Accuracy | FPS (Batch Size=8) | Input Resolution |
| ---- | ----- | ----------- | ------------------ | ---------------- |
|**Face Detection:** | scrfd_10g | 82.06 | 303.40 | 640x640x3 |
|**Face Recognition:**| arcface_mobilefacenet_v1 | 99.47 | 3457.79 | 112x112x3 |
|**Object Detection**| yolov8m | 49.10 | 139.10 | 640x640x3 |

## Test
Running the until tests:
```
cd /workspaces && \
colcon test --event-handlers console_direct+
```
The ros node will run on the container startup, stop it:
```
supervisorctl stop face_recognition
```
Running the end to end test:
```
ros2 launch hailo_face_recognition hailo.test.launch.py
```
## Run
The ros node will run on the container startup, if you want to stop it run:
```
supervisorctl stop face_recognition
```
If you want to run it directly:
```
ros2 launch hailo_face_recognition hailo.launch.py
```
Saving a detected face in the gallery
```
ros2 service call /hailo_face_recognition/save_face hailo_msgs/srv/SaveGalleryItem "{id: 'your_id', append: false}"
```
Deleting a face from the gallery
```
ros2 service call /hailo_face_recognition/delete_face hailo_msgs/srv/DeleteGalleryItem "{id: 'your_id'}"
```

# API

## Published topics

| Topic                                        | Type                                                                                                                                | Description                     |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| /hailo_face_recognition/image_bbx/compressed | [`sensor_msgs.msg.Image`](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/Image.msg)                           | image frame with bounding boxes |
| /hailo_face_recognition/detections           | [`vision_msgs.msg.Detection2DArray`](https://github.com/ros-perception/vision_msgs/blob/4.1.1/vision_msgs/msg/Detection2DArray.msg) | A list of 2D detections         |

## Exposed Services

These services allow you to manage the face gallery:

| Name                                | Type                                                         | Description                                                  |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| /hailo_face_recognition/save_face   | [`hailo_msgs/srv/SaveGalleryItem`](hailo_msgs/srv/SaveGalleryItem.srv)     | Adding or appending embedding from current detection to face |
| /hailo_face_recognition/delete_face | [`hailo_msgs/srv/DeleteGalleryItem`](hailo_msgs/srv/DeleteGalleryItem.srv) | Deleting all face embeddings from face |

## Configuration parameters

The default configuration file is [here](hailo_face_recognition/hailo_face_recognition/config/hailo_params.yaml).

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

| ROS2 Detection2D Type                      | Hailo Detection Type             |
| ------------------------------------------ | -------------------------------- |
| Detection2D.id                             | tracing_id                       |
| Detection2D.results[0].hypothesis.class_id | class_id + ": " + classification |
| Detection2D.bbox                           | hailo_bbox                       |

As you can see above ROS2 Detection2D have provisions for only the **Detection2D.id** and the **ObjectHypothesis.class_id** as such in this solution when a face is presented and regognised the **classification** is concatenated in the **ObjectHypothesis.class_id**. The **global_id** is not getting mapped anywhere.

### Performance

In the case of the host is engaging only to facilitate the models inference and running the GStreamer pipeline the CPU consumption is really low at around **20%** per core on RPi5.

When using the embeddings_gallery like in the face recognition use case the complexity is **O(N*M)** where **N** the total number of embeddings and **M** the number of face detections in a single frame. In short a RPi5 can handle around **200** embeddings with **5** faces on the same frame. This could be further improved by comparing the embeddings in a multithreading fashion and/or porting this code to C++ or Rust.

### ROS Domain ID

The ROS domain ID is set to #20. Feel free to change that to your needs.

# Acknowledgements
* https://github.com/hailo-ai/hailo-rpi5-examples
* https://github.com/hailo-ai/tappas
* https://github.com/canonical/pi-ai-kit-ubuntu
* https://github.com/Ar-Ray-code/rpi-bullseye-ros2
