# Hailo Raspberry Pi5 ROS2 
Face detection tappas pipeline publishing detection data and video to a ROS2 topic

![Face Recognition and Yolo](face_recognition.gif)

## Installation
```
docker compose build
docker compose up -d hailo-rpi-service
```
OR just open it with VSCode using the Dev Containers extension

## Test
Running the until tests:
```
cd /workspaces && \
colcon test --event-handlers console_direct+
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

### Acknowledgements
* https://github.com/hailo-ai/hailo-rpi5-examples
* https://github.com/canonical/pi-ai-kit-ubuntu
* https://github.com/Ar-Ray-code/rpi-bullseye-ros2