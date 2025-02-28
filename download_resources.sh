#!/bin/bash

# Set the resource directory
RESOURCE_DIR="./resources"
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
)

echo "Downloading HAILO8L models..."
for url in "${H8L_HEFS[@]}"; do
  download_model "$url" &
done

# Wait for all background downloads to complete
wait

echo "All downloads completed successfully!"
