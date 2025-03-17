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


import unittest
from unittest.mock import MagicMock, patch

import rclpy
from hailo_rpi_ros2_interfaces.srv import SaveFace, DeleteFace
from hailo_rpi_ros2.face_gallery import (
    Gallery,
    GalleryAppendStatus,
    GalleryDeletionStatus,
)
from hailo_rpi_ros2.face_recognition import FaceRecognition
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
)
import cv2

from hailo_rpi_ros2.face_recognition_node import HailoDetection


class TestHailoDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ros_args = [
            "--ros-args",
            "-p",
            "face_recognition.input:=test_input",
            "-p",
            "face_recognition.local_gallery_file:=test_gallery.json",
            "-p",
            "face_recognition.similarity_threshhold:=0.5",
            "-p",
            "face_recognition.queue_size:=10",
        ]
        rclpy.init(args=ros_args)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_add_face_callback_success(self):
        node = HailoDetection()
        node.gallery = MagicMock(spec=Gallery)
        node.gallery.append_new_item.return_value = GalleryAppendStatus.SUCCESS

        request = SaveFace.Request()
        request.name = "test_face"
        request.append = False

        response = node.add_face_callback(request, SaveFace.Response())

        self.assertEqual(response.result, 0)
        self.assertEqual(response.message, "Person added")
        node.gallery.append_new_item.assert_called_once_with("test_face", False)
        node.destroy_node()

    def test_add_face_callback_face_exists_with_identical_name(self):
        node = HailoDetection()
        node.gallery = MagicMock(spec=Gallery)
        node.gallery.append_new_item.return_value = (
            GalleryAppendStatus.FACE_EXISTS_WITH_IDENTICAL_NAME
        )

        request = SaveFace.Request()
        request.name = "test_face"
        request.append = False

        response = node.add_face_callback(request, SaveFace.Response())

        self.assertEqual(response.result, 1)
        self.assertEqual(
            response.message,
            (
                "Similar embedding found with identical name. "
                "Aborted Consider calling with append=true"
            )
        )
        node.destroy_node()

    def test_add_face_callback_multiple_faces_found(self):
        node = HailoDetection()
        node.gallery = MagicMock(spec=Gallery)
        node.gallery.append_new_item.return_value = (
            GalleryAppendStatus.MULTIPLE_FACES_FOUND
        )

        request = SaveFace.Request()
        request.name = "test_face"
        request.append = False

        response = node.add_face_callback(request, SaveFace.Response())

        self.assertEqual(response.result, 2)
        self.assertEqual(response.message, "Error: Multiple faces found")
        node.destroy_node()

    def test_delete_face_callback_success(self):
        node = HailoDetection()
        node.gallery = MagicMock(spec=Gallery)
        node.gallery.delete_item_by_name.return_value = (
            GalleryDeletionStatus.SUCCESS
        )

        request = DeleteFace.Request()
        request.name = "test_face"

        response = node.delete_face_callback(request, DeleteFace.Response())

        self.assertEqual(response.result, 0)
        self.assertEqual(response.message, "Face deleted")
        node.gallery.delete_item_by_name.assert_called_once_with("test_face")
        node.destroy_node()

    def test_delete_face_callback_not_found(self):
        node = HailoDetection()
        node.gallery = MagicMock(spec=Gallery)
        node.gallery.delete_item_by_name.return_value = (
            GalleryDeletionStatus.NOT_FOUND
        )

        request = DeleteFace.Request()
        request.name = "test_face"

        response = node.delete_face_callback(request, DeleteFace.Response())

        self.assertEqual(response.result, 1)
        self.assertEqual(response.message, "Name not found")
        node.destroy_node()

    @patch("hailo_rpi_ros2.face_recognition_pipeline.GStreamerFaceRecognitionApp")
    @patch("threading.Thread")
    def test_detection_callback(self, mock_thread, mock_gstreamer_app):
        node = HailoDetection()
        mock_gstreamer_instance = MagicMock()
        mock_gstreamer_app.return_value = mock_gstreamer_instance
        node.detections_publisher = MagicMock()
        mock_publisher = MagicMock()
        node.detections_publisher = mock_publisher
        node.get_clock = MagicMock()
        node.get_clock.now = MagicMock()
        node.get_clock.now.to_msg = MagicMock()

        detections = [Detection2D()]
        node.detection_callback(detections)
        mock_publisher.publish.assert_called_once()
        published_msg = mock_publisher.publish.call_args[0][0]
        self.assertIsInstance(published_msg, Detection2DArray)
        self.assertEqual(len(published_msg.detections), 1)
        node.destroy_node()

    @patch("hailo_rpi_ros2.face_recognition_pipeline.GStreamerFaceRecognitionApp")
    def test_frame_callback(self, mock_gstreamer_app):
        node = HailoDetection()
        mock_gstreamer_instance = MagicMock()
        mock_gstreamer_app.return_value = mock_gstreamer_instance
        node.image_publisher_compressed = MagicMock()
        node.get_clock = MagicMock()
        node.get_clock.now = MagicMock()
        node.get_clock.now.to_msg = MagicMock()

        frame = cv2.UMat(100, 100, cv2.CV_8UC3)
        node.frame_callback(frame)
        node.image_publisher_compressed.publish.assert_called_once()
        node.destroy_node()

    @patch("os.path.abspath")
    @patch("os.path.dirname")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="test")
    def test_get_absolute_file_path_in_build_dir_success(
        self, mock_open, mock_dirname, mock_abspath
    ):
        node = HailoDetection()
        mock_dirname.return_value = "/path/to/current/dir"
        mock_abspath.return_value = "/path/to/current/dir/hailo_detection.py"

        result = node._get_absolute_file_path_in_build_dir("test_file.json")
        self.assertEqual(result, "/path/to/current/dir/resources/test_file.json")
        node.destroy_node()

    @patch("os.path.abspath")
    @patch("os.path.dirname")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_get_absolute_file_path_in_build_dir_file_not_found(
        self, mock_open, mock_dirname, mock_abspath
    ):
        node = HailoDetection()
        mock_dirname.return_value = "/path/to/current/dir"
        mock_abspath.return_value = "/path/to/current/dir/hailo_detection.py"

        result = node._get_absolute_file_path_in_build_dir("test_file.json")
        with self.assertRaises(FileNotFoundError):
            node._get_absolute_file_path_in_build_dir("test_file.json")
        node.destroy_node()

if __name__ == '__main__':
    unittest.main()