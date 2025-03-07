import rclpy
from rclpy.node import Node
import hailo
from hailo_apps_infra.face_detection_pipeline import GStreamerFaceDetectionApp
from gi.repository import Gst
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
import face_gallery
import debugpy
from hailo_rpi_ros2.srv import AddPerson

class HailoDetection(Node, app_callback_class):
    def __init__(self):
        Node.__init__(self, 'hailo_detection')
        app_callback_class.__init__(self)

        self.new_variable = 42  # New variable example

        self.image_publisher_compressed = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)
        self.image_publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        self.srv = self.create_service(AddPerson, 'add_person', self.add_person_callback)

        self.gallery = face_gallery.Gallery(similarity_thr=0.4, queue_size=100)
        self.gallery.load_local_gallery_from_json('/workspaces/src/hailo-rpi-ros2/venv_hailo_rpi5_examples/lib/python3.11/site-packages/resources/face_recognition_local_gallery.json')

        app = GStreamerFaceDetectionApp(self.app_callback, self)
        app.run()
    
    def add_person_callback(self, request, response):
        self.get_logger().info(f'Incoming request: Add person {request.name}')
        
    def new_function(self):  # New function example
        return "The meaning of life is: "
    
    # This is the callback function that will be called when data is available from the pipeline
    def app_callback(self, pad, info, user_data):
        debugpy.debug_this_thread()
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
            bbox = detection.get_bbox()
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
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
        if user_data.use_frame:
            # Note: using imshow will not work here, as the callback function is not running in the main thread
            # Let's print the detection count to the frame
            cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Example of how to use the new_variable and new_function from the user_data
            # Let's print the new_variable and the result of the new_function to the frame
            cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Convert the frame to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            msg = CompressedImage()
            msg.header.frame_id = 'camera_frame'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = buffer.tobytes()

            self.image_publisher_compressed.publish(msg)

        print(string_to_print)
        return Gst.PadProbeReturn.OK

def main(args=None):
    rclpy.init(args=args)

    detection = HailoDetection()

    rclpy.spin(detection)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detection.destroy_node()
    rclpy.shutdown()

# Main program logic follows:
if __name__ == '__main__':
    main()