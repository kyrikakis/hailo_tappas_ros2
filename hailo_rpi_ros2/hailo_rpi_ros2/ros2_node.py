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
from hailo_rpi_ros2_interfaces.srv import AddPerson
from hailo_rpi_ros2 import face_recognition
from hailo_rpi_ros2 import face_gallery
import cv2
from rclpy import Parameter
from hailo_rpi_ros2.face_recognition_pipeline import GStreamerFaceRecognitionApp
from threading import Thread


class HailoDetection(Node):
    def __init__(self):
        Node.__init__(self, "hailo_detection")

        self.image_publisher_compressed = self.create_publisher(
            CompressedImage, "/camera/image_raw/compressed", 10
        )
        self.image_publisher_ = self.create_publisher(Image, "/camera/image_raw", 10)

        self.srv = self.create_service(
            AddPerson, "~/add_person", self.add_person_callback
        )

        self.declare_parameters(
            namespace="",
            parameters=[
                ("face_recognition.input", Parameter.Type.STRING),
                ("face_recognition.local_gallery_file", Parameter.Type.STRING),
                ("face_recognition.similarity_threshhold", Parameter.Type.DOUBLE),
                ("face_recognition.queue_size", Parameter.Type.INTEGER),
            ],
        )
        self.input = (
            self.get_parameter("face_recognition.input")
            .get_parameter_value()
            .string_value
        )
        self.local_gallery_file = (
            self.get_parameter("face_recognition.local_gallery_file")
            .get_parameter_value()
            .string_value
        )
        self.similarity_threshhold = (
            self.get_parameter("face_recognition.similarity_threshhold")
            .get_parameter_value()
            .double_value
        )
        self.queue_size = (
            self.get_parameter("face_recognition.queue_size")
            .get_parameter_value()
            .integer_value
        )

        gallery = face_gallery.Gallery(
            json_file_path=self.local_gallery_file,
            similarity_thr=self.similarity_threshhold,
            queue_size=self.queue_size,
        )

        self.face_recognition = face_recognition.FaceRecognition(
            gallery, self.frame_callback
        )

        gstreamer_app = GStreamerFaceRecognitionApp(
            self.input, self.face_recognition.app_callback, self.face_recognition
        )

        self.detection_thread = Thread(target=gstreamer_app.run)
        self.detection_thread.start()

    def add_person_callback(
        self, request: AddPerson.Request, response: AddPerson.Response
    ):
        self.get_logger().info(f"Incoming request: Add person {request.name}")
        response.success = True
        response.message = "Person added"
        return response

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

    detection = HailoDetection()

    rclpy.spin(detection)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    if hasattr(detection, "detection_thread") and detection.detection_thread.is_alive():
        detection.detection_thread.join()  # Wait for the thread to finish
        print("Detection thread joined.")
    detection.destroy_node()
    rclpy.shutdown()


# Main program logic follows:
if __name__ == "__main__":
    main()
