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
from hailo_rpi_ros2 import face_gallery
from hailo_rpi_ros2 import face_recognition
import gi
from gi.repository import Gst

gi.require_version("Gst", "1.0")


class MockUserData:
    def __init__(self):
        self.count = 0
        self.frame = None

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count

    def set_frame(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame


@pytest.fixture
def mock_gallery():
    return MagicMock(spec=face_gallery.Gallery)


@pytest.fixture
def mock_frame_callback():
    return MagicMock()


@pytest.fixture
def mock_pad():
    mock_pad = MagicMock()
    mock_caps = MagicMock()
    mock_structure = MagicMock()
    mock_structure.get_value.side_effect = ["RGB", 640, 480]  # Set side effects
    mock_caps.get_structure.return_value = mock_structure
    mock_pad.get_current_caps.return_value = mock_caps
    return mock_pad


@pytest.fixture
def mock_info():
    mock = MagicMock()
    mock_buffer = MagicMock()
    mock_map_info = MagicMock()
    dummy_data = b"\x00" * (640 * 480 * 3)
    mock_map_info.data = dummy_data
    mock_buffer.map.return_value = (True, mock_map_info)
    mock.get_buffer.return_value = mock_buffer  # correctly setup.
    return mock


@pytest.fixture
def mock_user_data():
    return MockUserData()


@pytest.fixture
def mock_get_roi():
    with patch("hailo.get_roi_from_buffer") as mock:
        mock_roi = MagicMock()
        mock_detection = MagicMock()
        mock_detection.get_label.return_value = "TestLabel"
        mock_detection.get_confidence.return_value = 0.95
        mock_classification = MagicMock()
        mock_classification.get_label.return_value = "PersonName"
        mock_detection.get_objects_typed.return_value = [mock_classification]
        mock_unique_id = MagicMock()
        mock_unique_id.get_id.return_value = 123
        mock_detection.get_objects_typed.side_effect = [
            [mock_classification],
            [mock_unique_id],
        ]
        mock_roi.get_objects_typed.return_value = [mock_detection]
        mock.return_value = mock_roi
        yield mock


def test_app_callback(
    mock_gallery,
    mock_frame_callback,
    mock_pad,
    mock_info,
    mock_user_data,
    mock_get_roi,
):
    face_recog = face_recognition.FaceRecognition(mock_gallery, mock_frame_callback)
    result = face_recog.app_callback(mock_pad, mock_info, mock_user_data)

    assert result == Gst.PadProbeReturn.OK
    assert mock_user_data.get_count() == 1
    mock_gallery.update.assert_called_once_with(
        [mock_get_roi.return_value.get_objects_typed.return_value[0]]
    )
    mock_frame_callback.assert_called_once()
    assert mock_user_data.get_frame() is not None


def test_app_callback_no_buffer(
    mock_gallery, mock_frame_callback, mock_pad, mock_info, mock_user_data, mock_get_roi
):
    mock_info.get_buffer.return_value = None
    face_recog = face_recognition.FaceRecognition(mock_gallery, mock_frame_callback)
    result = face_recog.app_callback(mock_pad, mock_info, mock_user_data)

    assert result == Gst.PadProbeReturn.OK
    assert mock_user_data.get_count() == 0
    mock_get_roi.return_value.assert_not_called()
    mock_frame_callback.assert_not_called()
