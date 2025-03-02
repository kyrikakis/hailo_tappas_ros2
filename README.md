# Hailo Raspberry Pi5 ROS2 
Face detection tappas pipeline publishing detection data and video to a ROS2 topic

## Installation
```
docker compose build
docker compose up -d hailo-rpi-service
```
OR just open it with VSCode using the Dev Containers extension
## Run
```
docker compose exec hailo-rpi-service /bin/bash
python3 src/ros2_node.py --use-frame --input rpi
```

### Acknowledgements
https://github.com/hailo-ai/hailo-rpi5-examples
https://github.com/canonical/pi-ai-kit-ubuntu