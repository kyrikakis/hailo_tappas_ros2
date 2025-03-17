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
from vision_msgs.msg import (
    BoundingBox2D,
)

gi.require_version("Gst", "1.0")


@pytest.fixture
def mock_gallery():
    return MagicMock(spec=face_gallery.Gallery)


@pytest.fixture
def mock_frame_callback():
    return MagicMock()


@pytest.fixture
def mock_detections_callback():
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
def mock_get_roi():
    with patch("hailo.get_roi_from_buffer") as mock:
        mock_roi = MagicMock()
        mock_detection = MagicMock()
        mock_detection.get_label.return_value = "TestLabel"
        mock_detection.get_confidence.return_value = 0.95
        mock_classification = MagicMock()
        mock_classification.get_label.return_value = "PersonName"
        mock_unique_id = MagicMock()
        mock_unique_id.get_id.return_value = 123
        mock_detection.get_objects_typed.side_effect = [
            [mock_classification],
            [mock_unique_id],
        ]
        mock_bbox = MagicMock()
        mock_bbox.xmin.return_value = 0.1
        mock_bbox.ymin.return_value = 0.2
        mock_bbox.width.return_value = 0.3
        mock_bbox.height.return_value = 0.4
        mock_detection.get_bbox.return_value = mock_bbox
        mock_roi.get_objects_typed.return_value = [mock_detection]
        mock.return_value = mock_roi
        yield mock


def test_app_callback(
    mock_gallery,
    mock_frame_callback,
    mock_detections_callback,
    mock_pad,
    mock_info,
    mock_get_roi,
):
    face_recog = face_recognition.FaceRecognition(
        mock_gallery, mock_frame_callback, mock_detections_callback
    )
    result = face_recog.app_callback(mock_pad, mock_info)

    assert result == Gst.PadProbeReturn.OK
    mock_gallery.update.assert_called_once_with(
        [mock_get_roi.return_value.get_objects_typed.return_value[0]]
    )
    mock_frame_callback.assert_called_once()
    mock_detections_callback.assert_called_once()

    # Assert Detection2D content
    detections_array = mock_detections_callback.call_args[0][0]
    assert isinstance(detections_array, list)
    assert len(detections_array) == 1

    detection = detections_array[0]
    assert detection.id == 123  # Check track ID

    # Check ObjectHypothesisWithPose
    hypothesis = detection.results[0].hypothesis
    assert hypothesis.class_id == "TestLabel: PersonName"
    assert hypothesis.score == 0.95

    # Check BoundingBox2D
    bbox = detection.bbox
    assert isinstance(bbox, BoundingBox2D)

    # Access the mocked hailo_bbox values
    hailo_bbox = mock_get_roi.return_value.get_objects_typed.return_value[
        0
    ].get_bbox.return_value
    xmin = hailo_bbox.xmin.return_value
    ymin = hailo_bbox.ymin.return_value
    width = hailo_bbox.width.return_value
    height = hailo_bbox.height.return_value

    assert bbox.center.position.x == xmin + width / 2.0
    assert bbox.center.position.y == ymin + height / 2.0
    assert bbox.size_x == width
    assert bbox.size_y == height
    assert isinstance(bbox.center.position.x, float)
    assert isinstance(bbox.center.position.y, float)
    assert isinstance(bbox.size_x, float)
    assert isinstance(bbox.size_y, float)

    # Check that there is only one hypothesis.
    assert len(detection.results) == 1


def test_app_callback_no_buffer(
    mock_gallery,
    mock_frame_callback,
    mock_detections_callback,
    mock_pad,
    mock_info,
    mock_get_roi,
):
    mock_info.get_buffer.return_value = None
    face_recog = face_recognition.FaceRecognition(
        mock_gallery, mock_frame_callback, mock_detections_callback
    )
    result = face_recog.app_callback(mock_pad, mock_info)

    assert result == Gst.PadProbeReturn.OK
    mock_get_roi.return_value.assert_not_called()
    mock_frame_callback.assert_not_called()
    mock_detections_callback.assert_not_called()
