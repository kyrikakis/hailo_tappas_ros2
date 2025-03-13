# Copyright 2025 Stefanos Kyrikakis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# !/usr/bin/env python3


import pytest
from unittest.mock import MagicMock, patch

# Import the class to be tested
from hailo_rpi_ros2.face_recognition import FaceRecognition
from hailo_rpi_ros2 import face_gallery


@pytest.fixture
def face_detection_instance():
    mock_frame_callback = MagicMock()
    with patch("argparse.ArgumentParser.parse_args", return_value=MagicMock()), patch(
        "hailo_rpi_ros2.face_recognition_pipeline." "GStreamerFaceRecognitionApp"
    ) as mock_gstreamer_app, patch(
        "hailo_rpi_ros2.gstreamer_app." "GStreamerApp.create_pipeline"
    ) as mock_base_create_pipeline, patch(
        "hailo_rpi_ros2.gstreamer_app." "GStreamerApp.run"
    ) as mock_base_run:
        mock_gstreamer_app_instance = mock_gstreamer_app.return_value
        mock_gstreamer_app_instance.run.return_value = None
        mock_gstreamer_app_instance.create_pipeline.return_value = None
        mock_base_create_pipeline.return_value = None
        mock_base_run.return_value = None
        face_detection = FaceRecognition(
            'rpi',
            face_gallery.Gallery(similarity_thr=0.4, queue_size=100),
            mock_frame_callback,
        )
    return face_detection, mock_frame_callback


@patch("hailo_rpi_ros2.face_gallery.Gallery")
@patch("hailo.get_roi_from_buffer")
@patch("cv2.putText")
@patch("cv2.cvtColor")
@patch("hailo_apps_infra.hailo_rpi_common.get_numpy_from_buffer")
@patch("hailo_apps_infra.hailo_rpi_common.get_caps_from_pad")
def test_app_callback(
    mock_get_caps,
    mock_get_numpy,
    mock_cvt_color,
    mock_put_text,
    mock_get_roi,
    mock_gallery,
    face_detection_instance,
):
    face_detection, mock_frame_callback = face_detection_instance

    # Configure the mocked gallery.
    mock_gallery_instance = mock_gallery.return_value
    mock_gallery_instance.load_local_gallery_from_json.return_value = None

    mock_info = MagicMock()
    mock_buffer = MagicMock()
    mock_info.get_buffer.return_value = mock_buffer

    # Configure the buffer.map mock
    mock_map_info = MagicMock()
    mock_buffer.map.return_value = (True, mock_map_info)

    mock_user_data = MagicMock()
    mock_user_data.get_count.return_value = 1
    mock_user_data.use_frame = True

    # Configure get_caps_from_pad to return a valid format string
    mock_format, mock_width, mock_height = "RGB", 640, 480
    mock_get_caps.return_value = (mock_format, mock_width, mock_height)

    mock_frame = MagicMock()
    mock_get_numpy.return_value = mock_frame

    mock_roi = MagicMock()
    mock_detection = MagicMock()
    mock_detection.get_label.return_value = "person"
    mock_detection.get_confidence.return_value = 0.9
    mock_detection.get_objects_typed.side_effect = [
        [mock_detection],  # HAILO_DETECTION
        [MagicMock()],  # HAILO_MATRIX
        [MagicMock()],  # HAILO_CLASSIFICATION
        [MagicMock()],  # HAILO_UNIQUE_ID
    ]
    mock_get_roi.return_value = mock_roi
    mock_roi.get_objects_typed.return_value
