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
from hailo_rpi_ros2 import (
    face_gallery
)
import hailo
from hailo_apps_infra.face_detection_pipeline import GStreamerFaceDetectionApp
from typing import Callable
from gi.repository import Gst
from threading import Thread
import cv2


class FaceDetection(app_callback_class):
    def __init__(self, frame_callback: Callable[[cv2.UMat], None]):
        app_callback_class.__init__(self)
        self.frame_callback = frame_callback
        self.gallery = face_gallery.Gallery(similarity_thr=0.4, queue_size=100)
        self.gallery.load_local_gallery_from_json(
            '/usr/local/lib/python3.11/dist-packages/resources/' +
            'face_recognition_local_gallery.json'
        )

        app = GStreamerFaceDetectionApp(self.app_callback, self)

        self.detection_thread = Thread(target=app.run)
        self.detection_thread.start()

    def __del__(self):
        """Destructor to ensure the thread is joined when the object is destroyed."""
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.detection_thread.join()  # Wait for the thread to finish
            print("Detection thread joined.")

    # This is the callback function that will be called when data is available from the pipeline
    def app_callback(self, pad, info, user_data):
        # Get the GstBuffer from the probe info
        buffer = info.get_buffer()
        # Check if the buffer is valid
        if buffer is None:
            return Gst.PadProbeReturn.OK

        # Using the user_data to count the number of frames
        user_data.increment()
        string_to_print = f"Frame count: {user_data.get_count()}\n"

        # Get the caps from the pad
        format, width, height = get_caps_from_pad(pad)

        # If the user_data.use_frame is set to True, we can get the video frame from the buffer
        frame = None
        if user_data.use_frame and format is not None and width is not None and height is not None:
            # Get video frame
            frame = get_numpy_from_buffer(buffer, format, width, height)

        # Get the detections from the buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        # Parse the detections
        detection_count = 0
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            # Get track ID
            track_id = 0
            embeddings = detection.get_objects_typed(hailo.HAILO_MATRIX)
            if len(embeddings) == 1:
                detections = [detection]
                self.gallery.update(detections)
                person_embeddings = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
                if len(person_embeddings) > 0:
                    print('person: ', person_embeddings[0].get_label())
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
        if user_data.use_frame:
            cv2.putText(frame, f"Detections: {detection_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Convert the frame to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)
            self.frame_callback(frame)

        print(string_to_print)
        return Gst.PadProbeReturn.OK
