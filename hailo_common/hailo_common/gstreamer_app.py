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

from gi.repository import Gst, GLib, GObject
from hailo_apps_infra.gstreamer_helper_pipelines import get_source_type

import setproctitle
import signal
import os
import gi
import threading
import sys
import cv2
import numpy as np

gi.require_version("Gst", "1.0")

try:
    from picamera2 import Picamera2
except ImportError:
    pass  # Available only on Pi OS


# -----------------------------------------------------------------------------------------------
# GStreamerApp class
# -----------------------------------------------------------------------------------------------
class GStreamerApp:
    def __init__(
        self,
        input: str,
        video_width: int = 1280,
        video_height: int = 720,
        video_fps: int = 15,
    ):
        # Set the process title
        setproctitle.setproctitle("Hailo Python App")

        # Set up signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.shutdown)

        # Initialize variables
        tappas_post_process_dir = os.environ.get("TAPPAS_POST_PROC_DIR", "")
        if tappas_post_process_dir == "":
            print(
                "TAPPAS_POST_PROC_DIR environment variable is not set."
                "Please set it to by sourcing setup_env.sh"
            )
            exit(1)
        self.postprocess_dir = tappas_post_process_dir
        self.video_source = input
        self.source_type = get_source_type(self.video_source)
        self.video_sink = "autovideosink"
        self.pipeline = None
        self.loop = None
        self.threads = []
        self.error_occurred = False
        self.pipeline_latency = 300  # milliseconds

        # Set Hailo parameters; these parameters should be set based on the model used
        self.batch_size = 1
        self.video_width = video_width
        self.video_height = video_height
        self.video_fps = video_fps
        self.video_format = "RGB"
        self.hef_path = None
        self.app_callback = None

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def create_pipeline(self):
        # Initialize GStreamer
        Gst.init(None)

        pipeline_string = self.get_pipeline_string()
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
        except Exception as e:
            print(f"Error creating pipeline: {e}", file=sys.stderr)
            sys.exit(1)

        # Create a GLib Main Loop
        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}", file=sys.stderr)
            self.error_occurred = True
            self.shutdown()
        # QOS
        elif t == Gst.MessageType.QOS:
            # Handle QoS message here
            qos_element = message.src.get_name()
            print(f"QoS message received from {qos_element}")
        return True

    def on_eos(self):
        if self.source_type == "file":
            # Seek to the start (position 0) in nanoseconds
            success = self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)
            if success:
                print("Video rewound successfully. Restarting playback...")
            else:
                print("Error rewinding the video.", file=sys.stderr)
        else:
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        print("Shutting down... Hit Ctrl-C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.NULL)
        GLib.idle_add(self.loop.quit)

    def get_pipeline_string(self):
        # This is a placeholder function that should be overridden by the child class
        return ""

    def dump_dot_file(self):
        print("Dumping dot file...")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        return False

    def run(self):
        # Add a watch for messages on the pipeline's bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        # Connect pad probe to the identity element
        identity = self.pipeline.get_by_name("identity_callback")
        if identity is None:
            print(
                "Warning: identity_callback element not found,"
                "add <identity name=identity_callback> in your"
                "pipeline where you want the callback to be called."
            )
        else:
            identity_pad = identity.get_static_pad("src")
            identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback)

        hailo_display = self.pipeline.get_by_name("hailo_display")
        if hailo_display is None:
            print(
                "Warning: hailo_display element not found,"
                "add <fpsdisplaysink name=hailo_display> to"
                "your pipeline to support fps display."
            )

        # Disable QoS to prevent frame drops
        disable_qos(self.pipeline)

        if self.source_type == "rpi":
            picam_thread = threading.Thread(
                target=picamera_thread,
                args=(
                    self.pipeline,
                    self.video_width,
                    self.video_height,
                    self.video_fps,
                    self.video_format,
                ),
            )
            self.threads.append(picam_thread)
            picam_thread.start()

        # Set the pipeline to PAUSED to ensure elements are initialized
        self.pipeline.set_state(Gst.State.PAUSED)

        # Set pipeline latency
        new_latency = (
            self.pipeline_latency * Gst.MSECOND
        )  # Convert milliseconds to nanoseconds
        self.pipeline.set_latency(new_latency)

        # Set pipeline to PLAYING state
        self.pipeline.set_state(Gst.State.PLAYING)

        # Run the GLib event loop
        self.loop.run()

        # Clean up
        try:
            self.pipeline.set_state(Gst.State.NULL)
            for t in self.threads:
                t.join()
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)
        finally:
            if self.error_occurred:
                print("Exiting with error...", file=sys.stderr)
                sys.exit(1)
            else:
                print("Exiting...")
                sys.exit(0)


def picamera_thread(
    pipeline, video_width, video_height, video_fps, video_format, picamera_config=None
):
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    print("appsrc properties: ", appsrc)
    # Initialize Picamera2
    with Picamera2() as picam2:
        if picamera_config is None:
            # Default configuration
            main = {"size": (video_width, video_height), "format": "RGB888"}
            lores = {"size": (video_width, video_height), "format": "RGB888"}
            controls = {"FrameRate": video_fps}
            config = picam2.create_preview_configuration(
                main=main, lores=lores, controls=controls
            )
        else:
            config = picamera_config
        # Configure the camera with the created configuration
        picam2.configure(config)
        # Update GStreamer caps based on 'lores' stream
        lores_stream = config["lores"]
        format_str = "RGB" if lores_stream["format"] == "RGB888" else video_format
        width, height = lores_stream["size"]
        print(
            f"Picamera2 configuration: width={width}, height={height}, format={format_str}"
        )
        appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, format={format_str}, width={width}, height={height}, "
                f"framerate={video_fps}/1, pixel-aspect-ratio=1/1"
            ),
        )
        picam2.start()
        frame_count = 0
        print("picamera_process started")
        while True:
            frame_data = picam2.capture_array("lores")
            # frame_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            if frame_data is None:
                print("Failed to capture frame.")
                break
            # Convert framontigue data if necessary
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            # Create Gst.Buffer by wrapping the frame data
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())
            # Set buffer PTS and duration
            buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buffer.pts = frame_count * buffer_duration
            buffer.duration = buffer_duration
            # Push the buffer to appsrc
            ret = appsrc.emit("push-buffer", buffer)
            if ret != Gst.FlowReturn.OK:
                print("Failed to push buffer:", ret)
                break
            frame_count += 1


def disable_qos(pipeline):
    """
    Iterate through all elements in the given GStreamer.

    pipeline and set the qos property to False
    where applicable.
    When the 'qos' property is set to True,
    the element will measure the time it takes to process
    each buffer and will drop frames if latency is too high.
    We are running on long pipelines, so we want to disable
    this feature to avoid dropping frames.
    :param pipeline: A GStreamer pipeline object
    """
    # Ensure the pipeline is a Gst.Pipeline instance
    if not isinstance(pipeline, Gst.Pipeline):
        print("The provided object is not a GStreamer Pipeline")
        return

    # Iterate through all elements in the pipeline
    it = pipeline.iterate_elements()
    while True:
        result, element = it.next()
        if result != Gst.IteratorResult.OK:
            break

        # Check if the element has the 'qos' property
        if "qos" in GObject.list_properties(element):
            # Set the 'qos' property to False
            element.set_property("qos", False)
            print(f"Set qos to False for {element.get_name()}")
