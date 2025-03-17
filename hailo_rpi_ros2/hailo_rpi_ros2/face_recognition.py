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


from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_rpi_ros2.face_gallery import (
    Gallery,
)
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
)
import hailo
from typing import (
    Callable,
    List,
)
from gi.repository import Gst
import gi
import cv2

# import debugpy

gi.require_version("Gst", "1.0")


class FaceRecognition(app_callback_class):
    def __init__(
        self,
        gallery: Gallery,
        frame_callback: Callable[[cv2.UMat], None],
        detections_callback: Callable[[Detection2DArray], None],
    ):
        app_callback_class.__init__(self)
        self.frame_callback = frame_callback
        self.detections_callback = detections_callback
        self.gallery = gallery

    # This is the callback function that will be called when data is available from the pipeline
    def app_callback(
        self, pad: gi.repository.Gst.Pad, info: gi.repository.Gst.PadProbeInfo
    ):
        # debugpy.debug_this_thread()
        # Get the GstBuffer from the probe info
        buffer = info.get_buffer()
        # Check if the buffer is valid
        if buffer is None:
            return Gst.PadProbeReturn.OK

        # Get the caps from the pad
        format, width, height = get_caps_from_pad(pad)

        frame = None
        if format is not None and width is not None and height is not None:
            # Get video frame
            frame = get_numpy_from_buffer(buffer, format, width, height)

        # Get the detections from the buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        self.gallery.update(detections)
        detections_2d = map_to_ros2_detection_2d_array(detections)
        self.detections_callback(detections_2d)
        if frame is not None:
            # Get frame
            cv2.putText(
                frame,
                f"Detections: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # Convert the frame to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.frame_callback(frame)
        return Gst.PadProbeReturn.OK


def map_to_ros2_detection_2d_array(
    detections: List[hailo.HAILO_DETECTION],
) -> List[Detection2D]:
    detections_2d = []
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        person_embeddings = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        if len(person_embeddings) > 0:
            label += f": {person_embeddings[0].get_label()}"
            print("person: ", person_embeddings[0].get_label())
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        detection_2d = Detection2D()
        detection_2d.id = track_id
        detection_2d.bbox = hailo_bbox_to_bounding_box_2d(detection.get_bbox())
        # Add ObjectHypothesisWithPose
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = label
        hypothesis.hypothesis.score = confidence
        detection_2d.results.append(hypothesis)
        detections_2d.append(detection_2d)
        print(
            f"Detection: ID: {track_id} "
            f"Label: {label} "
            f"Confidence: {confidence:.2f}\n"
        )
    return detections_2d


def hailo_bbox_to_bounding_box_2d(hailo_bbox: hailo.HailoBBox) -> BoundingBox2D:
    bounding_box_2d = BoundingBox2D()

    # Calculate center of the bounding box
    center_x = hailo_bbox.xmin() + hailo_bbox.width() / 2.0
    center_y = hailo_bbox.ymin() + hailo_bbox.height() / 2.0

    bounding_box_2d.center.position.x = center_x
    bounding_box_2d.center.position.y = center_y
    bounding_box_2d.center.theta = 0.0  # Assuming no rotation

    bounding_box_2d.size_x = hailo_bbox.width()
    bounding_box_2d.size_y = hailo_bbox.height()

    return bounding_box_2d
