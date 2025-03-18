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

from typing import Callable
from hailo_apps_infra.hailo_rpi_common import (
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
)
from hailo_common.gstreamer_app import (
    GStreamerApp,
)
import gi
import os
import setproctitle

gi.require_version("Gst", "1.0")


# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------


class GStreamerFaceRecognitionApp(GStreamerApp):
    def __init__(
        self,
        input: str,
        video_width: int,
        video_height: int,
        video_fps: int,
        object_detection: bool,
        app_callback: Callable[
            [gi.repository.Gst.Pad, gi.repository.Gst.PadProbeInfo], None
        ],
    ):
        # Call the parent class constructor
        super().__init__(input, video_width, video_height, video_fps)

        self.app_callback = app_callback
        self.object_detection = object_detection
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.vdevice_group_id = 1
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        self.current_path = os.path.dirname(os.path.abspath(__file__))
        infra_post_process_path = "/usr/local/lib/python3.11/dist-packages/resources/"
        tappas_post_process_path = (
            "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/"
        )

        detected_arch = detect_hailo_arch()
        if detected_arch is None:
            raise ValueError(
                "Could not auto-detect Hailo architecture. Please specify --arch manually."
            )
        self.arch = detected_arch
        print(f"Auto-detected Hailo architecture: {self.arch}")

        # Get models path

        # Set the HEF file path based on the arch
        if self.arch == "hailo8":
            self.yolo_hef_path = os.path.join(
                self.current_path, "resources/yolov8m.hef"
            )
        else:  # hailo8l
            self.yolo_hef_path = os.path.join(
                self.current_path, "resources/yolov8s_h8l.hef"
            )

        self.face_detection_hef_path = os.path.join(
            self.current_path, "resources/scrfd_10g.hef"
        )

        self.face_recognition_hef_path = os.path.join(
            self.current_path, "resources/arcface_mobilefacenet_v1.hef"
        )

        # Get post-process paths

        # Set the post-processing shared object file
        self.yolo_post_process_so = os.path.join(
            infra_post_process_path, "libyolo_hailortpp_postprocess.so"
        )
        self.yolo_post_function_name = "filter_letterbox"

        self.face_detection_post = os.path.join(
            tappas_post_process_path, "libscrfd_post.so"
        )
        self.face_detection_config = os.path.join(
            self.current_path, "resources/scrfd.json"
        )

        self.vms_cropper_so = os.path.join(
            tappas_post_process_path,
            "cropping_algorithms/libvms_croppers.so",
        )
        self.face_align_post = os.path.join(
            infra_post_process_path, "libvms_face_align.so"
        )
        self.face_recognition_post = os.path.join(
            tappas_post_process_path,
            "libface_recognition_post.so",
        )

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle("Hailo Detection App")

        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
            self.video_source, self.video_width, self.video_height
        )

        pipeline_string = (
            f"{source_pipeline} ! "
            f"{self.pre_detector_pipeline()}"
            f"{self.object_detection_pipeline()}"
            f"{self.face_detection_pipeline()}"
            f"{self.face_tracker_pipeline()}"
            f"{self.face_recognition_pipeline()}"
            f"{self.overlay_pipeline()}"
            f"{self.user_callback_pipeline()} ! "
            f"{self.display_pipeline()}"
        )
        print(pipeline_string)
        return pipeline_string

    def pre_detector_pipeline(self):
        pre_detection_pipeline = (
            "queue name=pre_detector_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            "tee name=t hailomuxer name=hmux t. ! "
            "queue name=detector_bypass_q leaky=no max-size-buffers=30 \
                max-size-bytes=0 max-size-time=0 ! "
            "hmux. t. ! "
            "videoscale name=face_videoscale method=0 n-threads=2 add-borders=false qos=false ! "
            "video/x-raw, pixel-aspect-ratio=1/1 ! "
        )
        return pre_detection_pipeline

    def object_detection_pipeline(self):
        object_detection_pipeline = ''
        if self.object_detection:
            object_detection_pipeline = (
                "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
                f"hailonet hef-path={self.yolo_hef_path} scheduling-algorithm=1 \
                    vdevice_group_id={self.vdevice_group_id} \
                        batch-size=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 \
                            output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
                "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
                f"hailofilter function-name={self.yolo_post_function_name} \
                    so-path={self.yolo_post_process_so} qos=false ! "
            )
        return object_detection_pipeline

    def face_detection_pipeline(self):
        face_detection_pipeline = (
            f"queue name=pre_face_detector_infer_q leaky=no max-size-buffers=30 \
                max-size-bytes=0 max-size-time=0 ! "
            f"hailonet hef-path={self.face_detection_hef_path} scheduling-algorithm=1 \
                vdevice_group_id={self.vdevice_group_id} ! "
            f"queue name=detector_post_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            f"hailofilter so-path={self.face_detection_post} \
                name=face_detection_hailofilter qos=false \
                    config-path={self.face_detection_config} function_name=scrfd_10g ! "
            f"queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hmux. hmux. ! "
        )
        return face_detection_pipeline

    def face_tracker_pipeline(self):
        tracker_pipeline = (
            "queue name=pre_tracker_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            "hailotracker name=hailo_face_tracker class-id=-1 kalman-dist-thr=0.7 \
                iou-thr=0.8 init-iou-thr=0.9 \
                    keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 \
                        keep-past-metadata=true qos=false ! "
            "queue name=hailo_post_tracker_q leaky=no max-size-buffers=30 \
                max-size-bytes=0 max-size-time=0 ! "
        )
        return tracker_pipeline

    def face_recognition_pipeline(self):
        inference_pipeline = (
            f"hailocropper so-path={self.vms_cropper_so} function-name=face_recognition \
                internal-offset=true name=cropper2 hailoaggregator name=agg2 cropper2. ! "
            f"queue name=bypess2_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! agg2. cropper2. ! "
            f"queue name=pre_face_align_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            f"hailofilter so-path={self.face_align_post} name=face_align_hailofilter \
                use-gst-buffer=true qos=false ! "
            f"queue name=detector_pos_face_align_q leaky=no max-size-buffers=30 \
                max-size-bytes=0 max-size-time=0 ! "
            f"hailonet hef-path={self.face_recognition_hef_path} scheduling-algorithm=1 \
                vdevice_group_id={self.vdevice_group_id} ! "
            f"queue name=recognition_post_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            f"hailofilter function-name=arcface_rgb so-path={self.face_recognition_post} \
                name=face_recognition_hailofilter qos=false ! "
            f"queue name=recognition_pre_agg_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! agg2. agg2. ! "
        )

        return inference_pipeline

    def overlay_pipeline(self):
        user_callback_pipeline = (
            "queue name=hailo_pre_draw2 leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            "hailooverlay name=hailo_overlay qos=false show-confidence=false local-gallery=false \
                line-thickness=5 font-thickness=2 landmark-point-radius=8 ! "
        )
        return user_callback_pipeline

    def user_callback_pipeline(self):
        user_callback_pipeline = (
            "queue name=dentity_callback_q leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            "identity name=identity_callback "
        )
        return user_callback_pipeline

    def display_pipeline(self):
        display_pipeline = (
            "queue name=hailo_post_draw leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            "videoconvert name=sink_videoconvert n-threads=2 qos=false ! "
            "queue name=hailo_display_q_0 leaky=no max-size-buffers=30 max-size-bytes=0 \
                max-size-time=0 ! "
            "fakevideosink sync=false "
        )
        return display_pipeline
