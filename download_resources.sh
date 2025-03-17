#!/bin/bash

# Set the resource directory
RESOURCE_DIR="hailo_rpi_ros2/hailo_rpi_ros2/resources"
mkdir -p "$RESOURCE_DIR"

# Define download function with file existence check and retries
download_model() {
  file_name=$(basename "$1")
  if [ ! -f "$RESOURCE_DIR/$file_name" ]; then
    echo "Downloading $file_name..."
    wget --tries=3 --retry-connrefused --quiet --show-progress "$1" -P "$RESOURCE_DIR" || {
      echo "Failed to download $file_name after multiple attempts."
      exit 1
    }
  else
    echo "File $file_name already exists. Skipping download."
  fi
}

# Define all URLs in arrays
H8_HEFS=(
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov8m.hef"
  "https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.31/general/hefs/arcface_mobilefacenet_v1.hef"
  "https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.31/general/hefs/scrfd_10g.hef"
)

H8L_HEFS=(
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8s_h8l.hef"
  "https://hailo-tappas.s3.eu-west-2.amazonaws.com/general/hefs/arcface_mobilefacenet_v1.hef"
  "https://hailo-tappas.s3.eu-west-2.amazonaws.com/general/hefs/scrfd_10g.hef"
)

FILES=(
  "https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.31/general/media/face_recognition.mp4"
  "https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.31/general/media/face_recognition/face_recognition_local_gallery.json"
)

echo "Downloading all models and video resources..."
for url in "${H8_HEFS[@]}" "${H8L_HEFS[@]}" "${FILES[@]}"; do
  download_model "$url" &
done

# Wait for all background downloads to complete
wait

echo "All downloads completed successfully!"
