import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    GStreamerApp,
    app_callback_class,
    dummy_callback,
    detect_hailo_arch,
)



# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to costume labels JSON file",
        )
        args = parser.parse_args()
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45


        # Determine the architecture if not specified
        if args.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = args.arch


        if args.hef_path is not None:
            self.hef_path = args.hef_path
        # Set the HEF file path based on the arch
        elif self.arch == "hailo8":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8m.hef')
        else:  # hailo8l
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')

        # Set the post-processing shared object file
        self.post_process_so = os.path.join(self.current_path, '../resources/libyolo_hailortpp_postprocess.so')

        # User-defined label JSON file
        self.labels_json = args.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle("Hailo Detection App")

        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(self.video_source)
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
        pipeline_string = (
            "gst-launch-1.0 libcamerasrc ! video/x-raw,format=YUY2,width=1024,height=576,framerate=15/1 ! queue max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! queue name=hailo_pre_convert_0 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! videoconvert n-threads=2 qos=false ! queue name=pre_detector_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! tee name=t hailomuxer name=hmux t. ! queue name=detector_bypass_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hmux. t. ! videoscale name=face_videoscale method=0 n-threads=2 add-borders=false qos=false ! video/x-raw, pixel-aspect-ratio=1/1 ! queue name=pre_face_detector_infer_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailonet hef-path=/home/pi/tappas/apps/h8/gstreamer/general/face_recognition/resources/scrfd_10g.hef scheduling-algorithm=1 vdevice-key=1 ! queue name=detector_post_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailofilter so-path=/home/pi/tappas/apps/h8/gstreamer/libs/post_processes//libscrfd_post.so name=face_detection_hailofilter qos=false config-path=/home/pi/tappas/apps/h8/gstreamer/general/face_recognition/resources/configs/scrfd.json function_name=scrfd_10g ! queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hmux. hmux. ! queue name=pre_tracker_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailotracker name=hailo_face_tracker class-id=-1 kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 keep-past-metadata=true qos=false ! queue name=hailo_post_tracker_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailocropper so-path=/home/pi/tappas/apps/h8/gstreamer/libs/post_processes//cropping_algorithms/libvms_croppers.so function-name=face_recognition internal-offset=true name=cropper2 hailoaggregator name=agg2 cropper2. ! queue name=bypess2_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg2. cropper2. ! queue name=pre_face_align_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailofilter so-path=/home/pi/tappas/apps/h8/gstreamer/libs/apps/vms//libvms_face_align.so name=face_align_hailofilter use-gst-buffer=true qos=false ! queue name=detector_pos_face_align_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailonet hef-path=/home/pi/tappas/apps/h8/gstreamer/general/face_recognition/resources/arcface_mobilefacenet_v1.hef scheduling-algorithm=1 vdevice-key=1 ! queue name=recognition_post_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailofilter function-name=arcface_rgb so-path=/home/pi/tappas/apps/h8/gstreamer/libs/post_processes//libface_recognition_post.so name=face_recognition_hailofilter qos=false ! queue name=recognition_pre_agg_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg2. agg2. ! queue name=hailo_pre_gallery_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailogallery gallery-file-path=/home/pi/tappas/apps/h8/gstreamer/general/face_recognition/resources/gallery/face_recognition_local_gallery_rgba.json load-local-gallery=true similarity-thr=.4 gallery-queue-size=20 class-id=-1 ! queue name=hailo_pre_draw2 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! hailofilter so-path=/home/pi/tappas/apps/h8/gstreamer/libs/post_processes/libros2_publisher.so qos=false ! queue ! hailooverlay name=hailo_overlay qos=false show-confidence=false local-gallery=true line-thickness=5 font-thickness=2 landmark-point-radius=8 ! queue name=hailo_post_draw leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! videoconvert n-threads=4 qos=false name=display_videoconvert qos=false ! queue name=hailo_display_q_0 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! jpegenc ! tcpserversink host=192.168.1.172"
            # f'{source_pipeline} '
            # f'{detection_pipeline} ! '
            # f'{user_callback_pipeline} ! '
            # f'{display_pipeline}'
        )
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
