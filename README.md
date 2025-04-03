# Hailo tappas ROS2

[![Version](https://img.shields.io/badge/version-1.0.3-green.svg)](https://github.com/kyrikakis/hailo_tappas_ros2/releases/tag/v1.0.3)

Designed for efficient deployment on the Raspberry Pi 5, this project delivers a pre-configured container integrating Hailo tappas and ROS2. It accelerates development with VS Code Dev Containers and ensures reliable production deployments through auto-restart and Supervisor as its process control system.

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

## Quick Start: Hailo Tappas with Docker Compose

To run the service:

```
docker compose up -d hailo-tappas-service
```

To access the container shell:

```
docker compose exec hailo-tappas-service /bin/bash
```

**Alternatively, use VS Code Dev Containers:**

Open this project in VS Code with the Dev Containers extension for a fully pre-configured development environment, bypassing manual Docker commands.

### Build image locally (Optional)

```
docker build -t ghcr.io/kyrikakis/hailo_tappas_ros2:v1.0.3 .
```

# Use Cases

## Face recognition

This ROS 2 node implements face recognition based on the **Hailo 8** tappas application example found [here](https://github.com/hailo-ai/tappas/blob/v3.31.0/apps/h8/gstreamer/general/face_recognition/README.rst). Building upon the original example, this node extends functionality by adding object detection alongside face detection and recognition. It publishes detection data and video streams to ROS2 topics and provides services for saving embeddings to a local gallery file.

Pushing the **Hailo 8** to **85%-95%** capacity, this node achieves high performance with relatively low host CPU consumption. A smooth **15 FPS** at **HD** video resolution is attainable, suitable for many applications. For increased frame rates, a parameter allows disabling object detection, potentially reaching up to **30 FPS** or more.

Furthermore, the original tappas [gallery](https://github.com/hailo-ai/tappas/blob/v3.31.0/core/hailo/plugins/gallery/gallery.hpp) has been modified, porting it to Python and enhancing its functionality. This includes support for multiple embeddings per person and the ability to add or remove embeddings during runtime. The modified gallery implementation can be found [here](hailo_tappas_ros2/hailo_common/hailo_common/embeddings_gallery.py).

![Face Recognition and Yolo](hailo_face_recognition/hailo_face_recognition/resources/face_recognition.gif)

## Models

| Task | Model | HW Accuracy | FPS (Batch Size=8) | Input Resolution |
| ---- | ----- | ----------- | ------------------ | ---------------- |
|**Face Detection** | scrfd_10g | 82.06 | 303.40 | 640x640x3 |
|**Face Recognition**| arcface_mobilefacenet_v1 | 99.47 | 3457.79 | 112x112x3 |
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

## API

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
| `Detection2D.id`                             | `tracing_id`                         |
| `Detection2D.results[0].hypothesis.class_id` | `class_id` + ": " + `classification` |
| `Detection2D.bbox`                           | `hailo_bbox`                         |

The ROS 2 `Detection2D` message provides fields for only two identifiers: `Detection2D.id` and `ObjectHypothesis.class_id`. Consequently, in this implementation, when a face is detected and recognized, the classification (e.g., person's name) is concatenated with the `class_id` within the `ObjectHypothesis.class_id` field. The `global_id`, representing persistent identity across application runs, is not directly mapped to a `Detection2D` field.

### Performance

When running the face recognition [test](https://github.com/kyrikakis/hailo_tappas_ros2/blob/main/hailo_face_recognition/hailo_face_recognition/launch/hailo.test.launch.py) with a small gallery of **10** embeddings and detecting **1-2** faces per frame, the Raspberry Pi 5's CPU usage remains remarkably low, averaging around **20%** per core. This indicates that the host's primary role is efficiently facilitating model inference and managing the GStreamer pipeline, with minimal overhead.

However, scaling up the system to handle larger face galleries introduces significant computational challenges. Comparing **N** stored embeddings against **M** detected faces results in a complexity of **O(N*M)**. It's important to note that a single person can be represented by multiple embeddings; for instance, **5** embeddings per face are often sufficient for robust recognition from various angles and lighting conditions. This means a gallery of **250** embeddings could represent **50** different individuals. This complexity becomes a bottleneck on the Raspberry Pi 5, limiting practical performance to **250** embeddings with **10** detected faces per frame. While this may be enought for many use cases in order to overcome this limitation and enable applications with larger galleries, we can significantly improve performance by employing multithreading or porting the code to a more efficient language like C++ or Rust.

### ROS Domain ID

The ROS domain ID is set to #20. Feel free to change that to your needs.

## Acknowledgements

* <https://github.com/hailo-ai/hailo-rpi5-examples>
* <https://github.com/hailo-ai/tappas>
* <https://github.com/canonical/pi-ai-kit-ubuntu>
* <https://github.com/Ar-Ray-code/rpi-bullseye-ros2>
