import numpy as np
import json
import hailo
import os
from typing import List, Dict, Tuple, Optional, Any

# Helper functions
def gallery_get_xtensor(matrix: np.ndarray) -> np.ndarray:
    return np.squeeze(matrix)

def gallery_one_dim_dot_product(array1, array2):
    """
    Calculates the dot product of two one-dimensional arrays.

    Args:
        array1: The first array.
        array2: The second array.

    Returns:
        The dot product of the two arrays.
    """
    array1 = np.array(array1)
    array2 = np.array(array2)

    if array1.ndim > 1 or array2.ndim > 1:
        raise RuntimeError("One of the arrays has more than 1 dimension")
    if array1.shape[0] != array2.shape[0]:
        raise RuntimeError("Arrays are with different shape")
    return np.sum(array1 * array2)

# Gallery class
class Gallery:
    def __init__(self, similarity_thr=0.15, queue_size=100):
        self.m_embeddings: List[List[np.ndarray]] = []
        self.tracking_id_to_global_id: Dict[int, int] = {}
        self.m_embedding_names: List[str] = []
        self.m_similarity_thr: float = similarity_thr
        self.m_queue_size: int = queue_size
        self.m_save_new_embeddings: bool = False
        self.m_json_file_path: Optional[str] = None
        self.m_load_local_embeddings: bool = False

    @staticmethod
    def get_distance(embeddings_queue: List[np.ndarray], matrix: np.ndarray) -> float:
        new_embedding = gallery_get_xtensor(matrix)
        max_thr = 0.0
        for embedding in embeddings_queue:
            thr = gallery_one_dim_dot_product(embedding, new_embedding)
            max_thr = max(thr, max_thr)
        return 1.0 - max_thr

    def get_embeddings_distances(self, matrix: np.ndarray) -> np.ndarray:
        distances = [self.get_distance(embeddings_queue, matrix) for embeddings_queue in self.m_embeddings]
        return np.array(distances)

    def init_local_gallery_file(self, file_path: str):
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump([], f)
        self.m_json_file_path = file_path
        self.m_save_new_embeddings = True

    def load_local_gallery_from_json(self, file_path: str):
        if not os.path.exists(file_path):
            raise RuntimeError("Gallery JSON file does not exist")
        self.m_json_file_path = file_path
        with open(file_path, "r") as f:
            data = json.load(f)
        roi = hailo.HailoROI(hailo.HailoBBox(0.0, 0.0, 1.0, 1.0))
        self.decode_hailo_face_recognition_result(data, roi, self.m_embedding_names)
        self.m_load_local_embeddings = True
        matrix_objs = roi.get_objects_typed(hailo.HAILO_MATRIX)
        for matrix in matrix_objs:
            global_id = self.create_new_global_id()
            self.add_embedding(global_id, matrix.get_data())

    def add_embedding(self, global_id: int, matrix: np.ndarray):
        global_id -= 1
        if len(self.m_embeddings[global_id]) >= self.m_queue_size:
            self.m_embeddings[global_id].pop()
        self.m_embeddings[global_id].insert(0, matrix)

    def encode_hailo_face_recognition_result(self, matrix: np.ndarray, name: str) -> dict:
        return {
            "name": name,
            "embedding": matrix.tolist()
        }

    def decode_hailo_face_recognition_result(self, object_json: list, roi: Any, embedding_names: list):
        """
        Decodes Hailo face recognition results from a list of dictionaries, with nested "FaceRecognition" data.

        Args:
            object_json: A list of dictionaries, where each dictionary may contain a "FaceRecognition" key.
            roi: The HailoROI object to which the decoded HailoMatrix objects will be added.
            embedding_names: A list to store the names of the recognized faces.
        """
        for entry in object_json:
            if isinstance(entry, dict) and "FaceRecognition" in entry:
                face_recognition = entry["FaceRecognition"]
                if "Embeddings" in face_recognition:
                    embedding_names.append(face_recognition.get("Name", "")) # get the name, if it does not exist, use ""
                    for embedding_entry in face_recognition["Embeddings"]:
                        if "HailoMatrix" in embedding_entry:
                            self.decode_matrix(embedding_entry["HailoMatrix"], roi)

    def decode_matrix(self, hailo_matrix_json: dict, roi: Any):
        """
        Decodes a HailoMatrix from a dictionary and adds it to the ROI.

        Args:
            hailo_matrix_json: A dictionary containing the HailoMatrix data.
            roi: The HailoROI object to which the decoded HailoMatrix object will be added.
        """
        if "data" in hailo_matrix_json and "height" in hailo_matrix_json and "width" in hailo_matrix_json and "features" in hailo_matrix_json:
            data = hailo_matrix_json["data"]
            height = hailo_matrix_json["height"]
            width = hailo_matrix_json["width"]
            features = hailo_matrix_json["features"]
            if isinstance(data, list) and isinstance(height, int) and isinstance(width, int) and isinstance(features, int):
                try:
                    # Flatten the data if it's not already flat
                    flat_data = []
                    if isinstance(data[0], list): #if the data is a 2d or 3d array.
                        for row in np.array(data).flatten():
                            flat_data.append(float(row))
                    else:
                        flat_data = [float(x) for x in data]

                    roi.add_object(hailo.HailoMatrix(flat_data, height, width, features))
                except ValueError:
                    print(f"Error decoding matrix with height: {height}, width: {width}, features: {features}")

    def write_to_json_file(self, document: dict):
        with open(self.m_json_file_path, "rb+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 2:
                f.seek(-1, os.SEEK_END)
                f.write(b",")
            else:
                f.seek(-1, os.SEEK_END)
            json.dump(document, f, indent=4)
            f.write(b"]")

    def save_embedding_to_json_file(self, matrix: np.ndarray, global_id: int):
        if self.m_save_new_embeddings:
            name = "Unknown" + str(global_id)
            self.write_to_json_file(self.encode_hailo_face_recognition_result(matrix, name))

    def create_new_global_id(self) -> int:
        self.m_embeddings.append([])
        return len(self.m_embeddings)

    def get_closest_global_id(self, matrix: np.ndarray) -> Tuple[int, float]:
        distances = self.get_embeddings_distances(matrix)
        closest_global_id = np.argmin(distances)
        return closest_global_id + 1, distances[closest_global_id]

    def get_embedding_matrix(self, detection: hailo.HailoDetection) -> Optional[np.ndarray]:
        embeddings = detection.get_objects_typed(hailo.HAILO_MATRIX)
        if not embeddings:
            return None
        elif len(embeddings) > 1:
            raise RuntimeError("A detection has more than 1 HailoMatrixPtr")
        return embeddings[0].get_data()

    def handle_local_embedding(self, detection: hailo.HailoDetection, global_id: int):
        if (global_id - 1) < len(self.m_embedding_names):
            classification_type = "recognition_result"
            existing_recognitions = [obj for obj in detection.get_objects_typed(hailo.HAILO_CLASSIFICATION) if obj.get_classification_type() == classification_type]
            name = self.m_embedding_names[global_id - 1]
            if not existing_recognitions:
                detection.add_object(hailo.HailoClassification(classification_type, name, 1.0))

    def update_embeddings_and_add_id_to_object(self, new_embedding: np.ndarray, detection: hailo.HailoDetection, global_id: int, unique_id: int):
        self.tracking_id_to_global_id[unique_id] = global_id
        if not self.m_load_local_embeddings and new_embedding is not None:
            self.add_embedding(global_id, new_embedding)
        global_ids = [obj for obj in detection.get_objects_typed(hailo.HAILO_UNIQUE_ID) if obj.get_mode() == hailo.GLOBAL_ID]
        if not global_ids:
            detection.add_object(hailo.HailoUniqueID(global_id, hailo.GLOBAL_ID))

    def new_embedding_to_global_id(self, new_embedding: np.ndarray, detection: hailo.HailoDetection, track_id: int):
        if track_id in self.tracking_id_to_global_id:
            # Global id to track already exists, add new embedding to global id
            self.update_embeddings_and_add_id_to_object(new_embedding, detection, self.tracking_id_to_global_id[track_id], track_id)
            if self.m_load_local_embeddings:
                self.handle_local_embedding(detection, self.tracking_id_to_global_id[track_id])
            return
        if new_embedding is None:
            # No embedding exists in this detection object, continue to next detection
            return
        if not self.m_embeddings:
            # Gallery is empty, adding new global id
            global_id = self.create_new_global_id()
            self.save_embedding_to_json_file(new_embedding, global_id)
            self.update_embeddings_and_add_id_to_object(new_embedding, detection, global_id, track_id)
            return
        # Get closest global id by distance between embeddings
        closest_global_id, min_distance = self.get_closest_global_id(new_embedding)
        if min_distance > self.m_similarity_thr:
            # if smallest distance is bigger than threshold and local gallery is not loaded -> create new global ID
            if not self.m_load_local_embeddings:
                global_id = self.create_new_global_id()
                self.save_embedding_to_json_file(new_embedding, global_id)
                self.update_embeddings_and_add_id_to_object(new_embedding, detection, global_id, track_id)
        else:
            # Close embedding found, update global id embeddings
            self.update_embeddings_and_add_id_to_object(new_embedding, detection, closest_global_id, track_id)
            if self.m_load_local_embeddings:
                self.handle_local_embedding(detection, closest_global_id)

    def update(self, detections: List[Any]): #replace Any with hailo.HailoDetection if needed.
        for detection in detections:
            track_ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if not track_ids:
                continue
            track_id = track_ids[0].get_id()
            new_embedding = self.get_embedding_matrix(detection)
            self.new_embedding_to_global_id(new_embedding, detection, track_id)

if __name__ == "__main__":
    # Example usage
    gallery = Gallery(similarity_thr=0.2, queue_size=100)

    # Initialize a local gallery file (if needed)
    # gallery.init_local_gallery_file("gallery.json")

    # Load local gallery from JSON (if needed)
    gallery.load_local_gallery_from_json("/workspaces/src/hailo-rpi-ros2/hailo_rpi_ros2/test_gallery.json")

    # Example HailoDetection and HailoMatrix (replace with your actual data)
    bbox1 = hailo.HailoBBox(0, 0, 10, 10)
    detection1 = hailo.HailoDetection(bbox1, "Person1", 0.9)

    matrix1 = hailo.HailoMatrix([0.7, 0.8, 0.9], 1, 1, 3)

    detection1.add_object(matrix1)

    # Corrected HailoUniqueID creation
    unique_id_object = hailo.HailoUniqueID(1)
    detection1.add_object(unique_id_object)

    detections = [detection1]

    # Update gallery with detections
    gallery.update(detections)

    person_embeddings = detections[0].get_objects_typed(hailo.HAILO_CLASSIFICATION)
    if len(person_embeddings) > 0:
        print('person: ', person_embeddings[0].get_label())
