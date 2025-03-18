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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from hailo_msgs.srv import (
    SaveGalleryItem,
    DeleteGalleryItem,
)
from hailo_face_recognition import face_recognition
from hailo_face_recognition.face_gallery import (
    Gallery,
    GalleryAppendStatus,
    GalleryDeletionStatus,
)
import cv2
from rclpy import Parameter
from hailo_face_recognition.face_recognition_pipeline import GStreamerFaceRecognitionApp
from threading import Thread
import os
from typing import (
    List,
)
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
)


class FaceRecognitionNode(Node):
    def __init__(self):
        Node.__init__(self, "hailo_face_recognition")

        self.image_publisher_compressed = self.create_publisher(
            CompressedImage, "~/image_bbx/compressed", 10
        )
        self.image_publisher = self.create_publisher(Image, "~/image_bbx", 10)

        self.detections_publisher = self.create_publisher(
            Detection2DArray, "~/detections", 10
        )

        self.create_service(SaveGalleryItem, "~/save_face", self.add_face_callback)

        self.create_service(
            DeleteGalleryItem, "~/delete_face", self.delete_face_callback
        )

        self.declare_parameters(
            namespace="",
            parameters=[
                ("input", Parameter.Type.STRING),
                ("local_gallery_file", Parameter.Type.STRING),
                ("similarity_threshhold", Parameter.Type.DOUBLE),
                ("video_width", Parameter.Type.INTEGER),
                ("video_height", Parameter.Type.INTEGER),
                ("video_fps", Parameter.Type.INTEGER),
                ("run_yolo", Parameter.Type.BOOL),
            ],
        )
        input = self.get_parameter("input").get_parameter_value().string_value
        local_gallery_file = (
            self.get_parameter("local_gallery_file").get_parameter_value().string_value
        )
        similarity_threshhold = (
            self.get_parameter("similarity_threshhold")
            .get_parameter_value()
            .double_value
        )
        video_width = (
            self.get_parameter("video_width").get_parameter_value().integer_value
        )
        video_height = (
            self.get_parameter("video_height").get_parameter_value().integer_value
        )
        video_fps = self.get_parameter("video_fps").get_parameter_value().integer_value
        run_yolo = self.get_parameter("run_yolo").get_parameter_value().bool_value

        gallery_file_path = self._get_absolute_file_path_in_build_dir(
            local_gallery_file
        )
        self.gallery = Gallery(
            json_file_path=gallery_file_path,
            similarity_thr=similarity_threshhold,
        )

        self.face_recognition = face_recognition.FaceRecognition(
            self.gallery, self.frame_callback, self.detection_callback
        )

        gstreamer_app = GStreamerFaceRecognitionApp(
            input,
            video_width,
            video_height,
            video_fps,
            run_yolo,
            self.face_recognition.app_callback,
        )

        self.detection_thread = Thread(target=gstreamer_app.run)
        self.detection_thread.start()

    def _get_absolute_file_path_in_build_dir(self, file: str) -> str:
        # Get the directory of the current Python file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute file path
        absolute_file_path = os.path.join(current_dir, "resources", file)
        return absolute_file_path

    def add_face_callback(
        self, request: SaveGalleryItem.Request, response: SaveGalleryItem.Response
    ):
        self.get_logger().info(f"Incoming request: Add face {request.id}")
        status = self.gallery.append_new_item(request.id, request.append)
        match status:
            case GalleryAppendStatus.SUCCESS:
                response.result = 0
                response.message = "Person added"
            case GalleryAppendStatus.SIMILAR_EMBEDDING_FOUND:
                response.result = 1
                response.message = (
                    "Similar embedding found with the same id. Aborted, "
                    "Consider calling with append=true"
                )
            case GalleryAppendStatus.SIMILAR_EMBEDDING_FOUND_WITH_DIFFERENT_ID:
                response.result = 1
                response.message = (
                    "Similar embedding found with different id. Aborted, "
                    "Consider calling with append=true"
                )
            case GalleryAppendStatus.ID_FOUND_WITH_DISTANT_EMBEDDING:
                response.result = 1
                response.message = (
                    "The id exists with distant embedding(s). Aborted, "
                    "Consider calling with append=true"
                )
            case GalleryAppendStatus.MULTIPLE_EMBEDDINGS_FOUND:
                response.result = 2
                response.message = "Error: Multiple faces found"
            case GalleryAppendStatus.NO_EMBEDDINGS_FOUND:
                response.result = 3
                response.message = "Error: No faces found"
            case _:
                response.result = 4
                response.message = "Failed, see the logs for more details"

        return response

    def delete_face_callback(
        self, request: DeleteGalleryItem.Request, response: DeleteGalleryItem.Response
    ):
        self.get_logger().info(f"Incoming request: Delete face {request.id}")
        status = self.gallery.delete_item_by_name(request.id)
        match status:
            case GalleryDeletionStatus.SUCCESS:
                response.result = 0
                response.message = "Face deleted"
            case GalleryDeletionStatus.NOT_FOUND:
                response.result = 1
                response.message = "Name not found"
            case _:
                response.result = 2
                response.message = "Failed, see the logs for more details"

        return response

    def detection_callback(self, detections: List[Detection2D]):
        msg = Detection2DArray()
        msg.header.frame_id = "camera_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.detections = detections

        for detection in msg.detections:
            detection.header.frame_id = "camera_frame"
            detection.header.stamp = msg.header.stamp

        self.detections_publisher.publish(msg)

    def frame_callback(self, frame: cv2.UMat):
        ret, buffer = cv2.imencode(".jpg", frame)
        msg = CompressedImage()
        msg.header.frame_id = "camera_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = buffer.tobytes()

        self.image_publisher_compressed.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    detection = FaceRecognitionNode()

    rclpy.spin(detection)
    if hasattr(detection, "detection_thread") and detection.detection_thread.is_alive():
        detection.detection_thread.join()  # Wait for the thread to finish
        print("Detection thread joined.")
    detection.destroy_node()
    rclpy.shutdown()


# Main program logic follows:
if __name__ == "__main__":
    main()
