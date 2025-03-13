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

import numpy as np
import json
import hailo
import os
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum


class GalleryUpdateStatus(Enum):
    SUCCESS = 0
    NO_TRACK_ID = 1
    NO_EMBEDDING = 2
    ITEM_ALREADY_EXISTS = 3


# Helper functions
def gallery_get_xtensor(matrix: np.ndarray) -> np.ndarray:
    return np.squeeze(matrix)


def gallery_one_dim_dot_product(array1, array2):
    """
    Calculate the dot product of two one-dimensional arrays.

    Args:
        array1 (list or numpy.ndarray): The first array.
        array2 (list or numpy.ndarray): The second array.

    Returns
    -------
        float or numpy.ndarray: The dot product of the two arrays.

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
    def __init__(
        self,
        similarity_thr=0.40,
        queue_size=100,
        m_json_file_path="face_recognition_local_gallery.json",
    ):
        self.m_embeddings: List[List[np.ndarray]] = []
        self.tracking_id_to_global_id: Dict[int, int] = {}
        self.m_embedding_names: List[str] = []
        self.m_similarity_thr: float = similarity_thr
        self.m_queue_size: int = queue_size
        self.m_json_file_path: Optional[str] = m_json_file_path

    @staticmethod
    def _get_distance(embeddings_queue: List[np.ndarray], matrix: np.ndarray) -> float:
        new_embedding = gallery_get_xtensor(matrix)
        max_thr = 0.0
        for embedding in embeddings_queue:
            thr = gallery_one_dim_dot_product(embedding, new_embedding)
            max_thr = max(thr, max_thr)
        return 1.0 - max_thr

    def _get_embeddings_distances(self, matrix: np.ndarray) -> np.ndarray:
        distances = [
            self._get_distance(embeddings_queue, matrix)
            for embeddings_queue in self.m_embeddings
        ]
        return np.array(distances)

    def _add_embedding(self, global_id: int, matrix: np.ndarray):
        global_id -= 1
        if len(self.m_embeddings[global_id]) >= self.m_queue_size:
            self.m_embeddings[global_id].pop()
        self.m_embeddings[global_id].insert(0, matrix)

    def _encode_hailo_face_recognition_result(
        self, matrix: np.ndarray, name: str
    ) -> dict:
        return {
            "FaceRecognition": {
                "Name": name,
                "Embeddings": [
                    {
                        "HailoMatrix": {
                            "width": 1,
                            "height": 1,
                            "features": matrix.shape[0],
                            "data": matrix.tolist(),
                        }
                    }
                ],
            }
        }

    def _decode_hailo_face_recognition_result(
        self, object_json: list, roi: Any, embedding_names: list
    ):
        for entry in object_json:
            if isinstance(entry, dict) and "FaceRecognition" in entry:
                face_recognition = entry["FaceRecognition"]
                if "Embeddings" in face_recognition:
                    # get the name, if it does not exist, use ""
                    embedding_names.append(face_recognition.get("Name", ""))
                    embeddings = face_recognition["Embeddings"]
                    for embedding_entry in embeddings:
                        if "HailoMatrix" in embedding_entry:
                            self._decode_matrix(embedding_entry["HailoMatrix"], roi)

    def _decode_matrix(self, hailo_matrix_json: dict, roi: Any):
        if (
            "data" in hailo_matrix_json
            and "height" in hailo_matrix_json
            and "width" in hailo_matrix_json
            and "features" in hailo_matrix_json
        ):

            data = hailo_matrix_json["data"]
            height = hailo_matrix_json["height"]
            width = hailo_matrix_json["width"]
            features = hailo_matrix_json["features"]
            if (
                isinstance(data, list)
                and isinstance(height, int)
                and isinstance(width, int)
                and isinstance(features, int)
            ):
                try:
                    # Flatten the data if it's not already flat
                    flat_data = []
                    # if the data is a 2d or 3d array.
                    if isinstance(data[0], list):
                        for row in np.array(data).flatten():
                            flat_data.append(float(row))
                    else:
                        flat_data = [float(x) for x in data]

                    roi.add_object(
                        hailo.HailoMatrix(flat_data, height, width, features)
                    )
                except ValueError:
                    print(
                        f"Error decoding matrix with height: {height}, "
                        f"width: {width}, features: {features}"
                    )

    def _write_to_json_file(self, document: dict):
        # Create the file if it doesn't exist
        if not os.path.exists(self.m_json_file_path):
            with open(self.m_json_file_path, "w") as f:
                json.dump([], f)  # Initialize with an empty list

        with open(self.m_json_file_path, "rb+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 2:
                f.seek(-1, os.SEEK_END)
                f.write(b",")
            else:
                f.seek(-1, os.SEEK_END)
            f.write(json.dumps(document, indent=4).encode("utf-8"))
            f.write(b"]")

    def _save_embedding_to_json_file(
        self, name: str, matrix: np.ndarray, global_id: int
    ):
        self._write_to_json_file(
            self._encode_hailo_face_recognition_result(matrix, name)
        )

    def _append_embedding_in_json(
        self, old_name: str, new_name: str, new_embedding: np.ndarray
    ):
        if not os.path.exists(self.m_json_file_path):
            raise FileNotFoundError(f"JSON file not found: {self.m_json_file_path}")

        try:
            with open(self.m_json_file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {self.m_json_file_path}")

        found = False
        for item in data:
            if (
                "FaceRecognition" in item
                and item["FaceRecognition"].get("Name") == old_name
            ):
                item["FaceRecognition"]["Name"] = new_name
                if new_embedding is not None:
                    new_embedding_data = {
                        "HailoMatrix": {
                            "width": 1,
                            "height": 1,
                            "features": new_embedding.shape[0],
                            "data": new_embedding.tolist()
                        }
                    }
                    item["FaceRecognition"]["Embeddings"].append(new_embedding_data)

                found = True
                break

        if not found:
            raise ValueError(f"Name '{old_name}' not found in JSON data.")

        try:
            with open(self.m_json_file_path, "w") as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            raise IOError(f"Error writing to JSON file: {e}")

    def _create_new_global_id(self) -> int:
        self.m_embeddings.append([])
        return len(self.m_embeddings)

    def _get_closest_global_id(self, matrix: np.ndarray) -> Tuple[int, float]:
        distances = self._get_embeddings_distances(matrix)
        closest_global_id = np.argmin(distances)
        return closest_global_id + 1, distances[closest_global_id]

    def _get_embedding_matrix(
        self, detection: hailo.HailoDetection
    ) -> Optional[np.ndarray]:
        embeddings = detection.get_objects_typed(hailo.HAILO_MATRIX)
        if not embeddings:
            return None
        elif len(embeddings) > 1:
            raise RuntimeError("A detection has more than 1 HailoMatrixPtr")
        return embeddings[0].get_data()

    def _handle_local_embedding(self, detection: hailo.HailoDetection, global_id: int):
        if (global_id - 1) < len(self.m_embedding_names):
            classification_type = "recognition_result"
            existing_recognitions = [
                obj
                for obj in detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
                if obj.get_classification_type() == classification_type
            ]
            name = self.m_embedding_names[global_id - 1]
            if not existing_recognitions:
                detection.add_object(
                    hailo.HailoClassification(classification_type, name, 1.0)
                )

    def _update_embeddings_and_add_id_to_object(
        self,
        new_embedding: np.ndarray,
        detection: hailo.HailoDetection,
        global_id: int,
        unique_id: int,
    ):
        self.tracking_id_to_global_id[unique_id] = global_id
        self._add_embedding(global_id, new_embedding)
        global_ids = [
            obj
            for obj in detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if obj.get_mode() == hailo.GLOBAL_ID
        ]
        if not global_ids:
            detection.add_object(hailo.HailoUniqueID(global_id, hailo.GLOBAL_ID))

    def _new_embedding_to_global_id(
        self, new_embedding: np.ndarray, detection: hailo.HailoDetection, track_id: int
    ):
        if track_id in self.tracking_id_to_global_id:
            # Global id to track already exists, add new embedding to global id
            self._update_embeddings_and_add_id_to_object(
                new_embedding,
                detection,
                self.tracking_id_to_global_id[track_id],
                track_id,
            )
            self._handle_local_embedding(
                detection, self.tracking_id_to_global_id[track_id]
            )
            return
        # Get closest global id by distance between embeddings
        closest_global_id, min_distance = self._get_closest_global_id(new_embedding)
        if min_distance < self.m_similarity_thr:
            # Close embedding found, update global id embeddings
            self._update_embeddings_and_add_id_to_object(
                new_embedding, detection, closest_global_id, track_id
            )
            self._handle_local_embedding(detection, closest_global_id)

    def _add_and_save_embedding(
        self,
        name: str,
        new_embedding: np.ndarray,
        detection: hailo.HailoDetection,
        track_id: int,
    ):
        global_id = self._create_new_global_id()
        self.m_embedding_names.append(name)
        self._save_embedding_to_json_file(name, new_embedding, global_id)
        self._update_embeddings_and_add_id_to_object(
            new_embedding, detection, global_id, track_id
        )
        self._handle_local_embedding(detection, global_id)

    def load_local_gallery_from_json(self, file_path: str):
        if not os.path.exists(file_path):
            raise RuntimeError("Gallery JSON file does not exist")
        self.m_json_file_path = file_path
        with open(file_path, "r") as f:
            data = json.load(f)
        roi = hailo.HailoROI(hailo.HailoBBox(0.0, 0.0, 1.0, 1.0))
        self._decode_hailo_face_recognition_result(data, roi, self.m_embedding_names)
        matrix_objs = roi.get_objects_typed(hailo.HAILO_MATRIX)
        for matrix in matrix_objs:
            global_id = self._create_new_global_id()
            self._add_embedding(global_id, matrix.get_data())

    def update(self, detections: List[hailo.HailoDetection]):
        for detection in detections:
            track_ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if not track_ids:
                continue
            track_id = track_ids[0].get_id()
            new_embedding = self._get_embedding_matrix(detection)
            if new_embedding is None:
                # No embedding exists in this detection object, continue to next detection
                continue
            self._new_embedding_to_global_id(new_embedding, detection, track_id)

    def register_new_item(
        self, name: str, detection: hailo.HailoDetection, append: bool
    ) -> GalleryUpdateStatus:
        track_ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if not track_ids:
            return GalleryUpdateStatus.NO_TRACK_ID
        track_id = track_ids[0].get_id()
        new_embedding = self._get_embedding_matrix(detection)
        if new_embedding is None:
            # No embedding exists in this detection object, continue to next detection
            return GalleryUpdateStatus.NO_EMBEDDING
        if not self.m_embeddings:
            # Gallery is empty, adding new global id
            self._add_and_save_embedding(name, new_embedding, detection, track_id)
            return GalleryUpdateStatus.SUCCESS
        # Check if there is no other object in close distance
        closest_global_id, min_distance = self._get_closest_global_id(new_embedding)
        if min_distance > self.m_similarity_thr:
            # if smallest distance is bigger than similarity threshold
            self._add_and_save_embedding(name, new_embedding, detection, track_id)
            return GalleryUpdateStatus.SUCCESS
        else:
            # Similar embedding found
            if append:
                old_name = self.m_embedding_names[closest_global_id - 1]
                self.m_embedding_names[closest_global_id - 1] = name
                self._append_embedding_in_json(old_name, name, new_embedding)
                self._update_embeddings_and_add_id_to_object(
                    new_embedding, detection, closest_global_id, track_id
                )
                self._handle_local_embedding(detection, closest_global_id)
                return GalleryUpdateStatus.SUCCESS
            else:
                return GalleryUpdateStatus.ITEM_ALREADY_EXISTS

    def delete_item(self, name: str):
        # You are never get removed, just removes your name
        self.m_embedding_names[self.m_embedding_names.index(name)] = ""
        self._append_embedding_in_json(name, "", None)
