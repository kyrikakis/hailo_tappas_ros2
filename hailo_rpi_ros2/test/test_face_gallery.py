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

from pathlib import Path
from typing import Any
import pytest
import numpy as np
from hailo_rpi_ros2.face_gallery import (
    Gallery,
    GalleryAppendStatus,
    GalleryDeletionStatus,
    gallery_one_dim_dot_product,
    gallery_get_xtensor,
)
from unittest.mock import MagicMock
import hailo
import json
import os

# Test helper functions


def test_gallery_one_dim_dot_product():
    array1 = [1, 2, 3]
    array2 = [4, 5, 6]
    result = gallery_one_dim_dot_product(array1, array2)
    assert result == 32  # 1*4 + 2*5 + 3*6 = 32


def test_gallery_one_dim_dot_product_invalid_dimensions():
    array1 = [[1, 2], [3, 4]]
    array2 = [1, 2, 3]
    with pytest.raises(RuntimeError):
        gallery_one_dim_dot_product(array1, array2)


def test_gallery_one_dim_dot_product_different_shapes():
    array1 = [1, 2, 3]
    array2 = [1, 2]
    with pytest.raises(RuntimeError):
        gallery_one_dim_dot_product(array1, array2)


def test_gallery_get_xtensor():
    matrix = np.array([[1, 2, 3]])
    result = gallery_get_xtensor(matrix)
    assert np.array_equal(result, np.array([1, 2, 3]))


# Test Gallery class


def test_gallery_initialization():
    gallery = Gallery(similarity_thr=0.2, queue_size=100)
    assert gallery.m_similarity_thr == 0.2
    assert gallery.m_queue_size == 100
    assert len(gallery.m_embeddings) == 0


def test_gallery_add_embedding():
    gallery = Gallery()
    global_id = gallery._create_new_global_id()
    matrix = np.array([1, 2, 3])
    gallery._add_embedding(global_id, matrix)
    assert len(gallery.m_embeddings[global_id - 1]) == 1
    assert np.array_equal(gallery.m_embeddings[global_id - 1][0], matrix)


def test_gallery_get_closest_global_id():
    gallery = Gallery()

    # Generate realistic embeddings
    matrix1 = generate_realistic_embedding()
    matrix2 = generate_realistic_embedding()
    matrix3 = generate_realistic_embedding()

    global_id1 = gallery._create_new_global_id()
    gallery._add_embedding(global_id1, matrix1)

    global_id2 = gallery._create_new_global_id()
    gallery._add_embedding(global_id2, matrix2)

    global_id3 = gallery._create_new_global_id()
    gallery._add_embedding(global_id3, matrix3)

    closest_id, distance = gallery._get_closest_global_id(matrix1)

    assert closest_id == global_id1

    # Add an extra assertion to check if a realistic distance is returned.
    # The distance should be small when comparing the same face.
    assert distance < 0.2

    # Test with a matrix very similar to matrix 2.
    matrix4 = matrix2 + np.random.normal(loc=0.0, scale=0.01, size=512)
    closest_id2, distance2 = gallery._get_closest_global_id(matrix4)
    assert closest_id2 == global_id2
    assert distance2 < 0.1


def test_gallery_get_closest_global_id_2_embedding_global():
    gallery = Gallery()

    # Generate realistic embeddings
    matrix1 = generate_realistic_embedding()
    matrix2 = generate_realistic_embedding()
    matrix3 = generate_realistic_embedding()

    global_id1 = gallery._create_new_global_id()
    gallery._add_embedding(global_id1, matrix1)

    global_id2 = gallery._create_new_global_id()
    gallery._add_embedding(global_id2, matrix2)
    gallery._add_embedding(global_id2, matrix3)

    closest_id, distance = gallery._get_closest_global_id(matrix3)

    assert closest_id == global_id2

    # Add an extra assertion to check if a realistic distance is returned.
    # The distance should be small when comparing the same face.
    assert distance < 0.2

    # Test with a matrix very similar to matrix 2.
    matrix4 = matrix2 + np.random.normal(loc=0.0, scale=0.01, size=512)
    closest_id2, distance2 = gallery._get_closest_global_id(matrix4)
    assert closest_id2 == global_id2
    assert distance2 < 0.1


# Helper function to generate realistic embeddings
def generate_realistic_embedding(size=512) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=0.1, size=size)


@pytest.fixture
def alon_face_matrix():
    yield generate_realistic_embedding()


@pytest.fixture
def max_face_matrix():
    yield generate_realistic_embedding()


# Fixture for the Gallery object and test JSON file
@pytest.fixture
def gallery_and_json(tmp_path: Path, alon_face_matrix: np.ndarray, max_face_matrix: np.ndarray):
    # Create a test JSON file
    test_json_path = tmp_path / "test_gallery.json"
    test_data = [
        {
            "FaceRecognition": {
                "Name": "Alon",
                "Embeddings": [
                    {
                        "HailoMatrix": {
                            "width": 1,
                            "height": 1,
                            "features": 512,
                            "data": alon_face_matrix.tolist(),
                        }
                    }
                ],
            }
        },
        {
            "FaceRecognition": {
                "Name": "Max",
                "Embeddings": [
                    {
                        "HailoMatrix": {
                            "width": 1,
                            "height": 1,
                            "features": 512,
                            "data": max_face_matrix.tolist(),
                        }
                    }
                ],
            }
        },
    ]
    with open(test_json_path, "w") as f:
        json.dump(test_data, f)

    # Initialize Gallery and load the test JSON
    gallery = Gallery(json_file_path=test_json_path)

    yield gallery, test_json_path

    # Clean up the test JSON file after the test
    os.remove(test_json_path)


# Test Case 1: Track ID already exists (and local embeddings loaded)
def test_alon_face_found(gallery_and_json: tuple[Gallery, Any], alon_face_matrix: np.ndarray):
    gallery, _ = gallery_and_json

    # Mock detection with existing track ID
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = alon_face_matrix
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 1

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [],  # No classifications
        [],  # Additional call (if any)
    ]

    # Call the public update method
    gallery.update([mock_detection])

    # Assertions
    calls = mock_detection.add_object.call_args_list
    assert len(calls) == 2
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 0
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert isinstance(calls[1][0][0], hailo.HailoClassification)
    assert calls[1][0][0].get_label() == "Alon"


def test_max_face_found(gallery_and_json: tuple[Gallery, Any], max_face_matrix: np.ndarray):
    gallery, _ = gallery_and_json

    # Mock detection with existing track ID
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = max_face_matrix
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 1

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [],  # No classifications
        [],  # Additional call (if any)
    ]

    # Call the public update method
    gallery.update([mock_detection])

    # Assertions
    calls = mock_detection.add_object.call_args_list
    assert len(calls) == 2
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 1
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert isinstance(calls[1][0][0], hailo.HailoClassification)
    assert calls[1][0][0].get_label() == "Max"


def test_update_no_embedding(gallery_and_json: tuple[Gallery, Any]):
    gallery, _ = gallery_and_json

    # Mock detection with no embedding
    mock_detection = MagicMock()
    mock_detection.get_objects_typed.return_value = []  # No embedding or unique ID

    # Call the public update method
    gallery.update([mock_detection])

    # Assertions
    mock_detection.add_object.assert_not_called()


def test_update_with_no_json_provided():
    gallery = Gallery()

    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = generate_realistic_embedding()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [],  # No classifications
        [],  # Additional call (if any)
    ]

    # Call the public update method
    gallery.update([mock_detection])

    # Assertions
    mock_detection.add_object.assert_not_called()


def test_update_dissimilar_embedding(gallery_and_json: tuple[Gallery, Any]):
    gallery, _ = gallery_and_json

    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = generate_realistic_embedding()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 4

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [],  # No classifications
        [],  # Additional call (if any)
    ]

    # Call the public update method
    gallery.update([mock_detection])

    # Assertions
    mock_detection.add_object.assert_not_called()
    assert 4 not in gallery.tracking_id_to_global_id


def test_add_item_to_existing_gallery(gallery_and_json: tuple[Gallery, Any]):
    gallery, test_json_path = gallery_and_json
    embedding = generate_realistic_embedding()
    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = embedding.tolist()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [],  # Empty existing classication
    ]

    gallery.update([mock_detection])

    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.SUCCESS
    )

    # Assertions
    calls = mock_detection.add_object.call_args_list
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 2
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert gallery.tracking_id_to_global_id[11] == 2

    # Check if the file was created and written to
    assert os.path.exists(str(test_json_path))
    with open(str(test_json_path), "r") as f:
        data = json.load(f)
    assert len(data) == 3
    assert data[2]["FaceRecognition"]["Name"] == "Stefanos"
    assert (
        data[2]["FaceRecognition"]["Embeddings"][0]["HailoMatrix"]["data"]
        == embedding.tolist()
    )


def test_add_two_items_to_empty_gallery(tmp_path: Path):
    gallery = Gallery(json_file_path=str(tmp_path / "test.json"))

    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = generate_realistic_embedding().tolist()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [],  # No classifications
    ]

    gallery.update([mock_detection])
    gallery.append_new_item(name="Stefanos", append=False)

    # Mock detection with dissimilar embedding
    mock_detection2 = MagicMock()
    mock_matrix2 = MagicMock()
    mock_matrix2.get_data.return_value = generate_realistic_embedding().tolist()
    mock_unique_id2 = MagicMock()
    mock_unique_id2.get_id.return_value = 12

    # Set up mock behavior
    mock_detection2.get_objects_typed.side_effect = [
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [],  # No classifications
    ]

    gallery.update([mock_detection2])
    gallery.append_new_item(name="Max", append=False)

    # Assertions
    calls = mock_detection.add_object.call_args_list
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 0
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert gallery.tracking_id_to_global_id[11] == 0

    calls = mock_detection2.add_object.call_args_list
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 1
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert gallery.tracking_id_to_global_id[12] == 1

    # Check if the file was created and written to
    assert os.path.exists(str(tmp_path / "test.json"))
    with open(str(tmp_path / "test.json"), "r") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["FaceRecognition"]["Name"] == "Stefanos"
    assert data[1]["FaceRecognition"]["Name"] == "Max"


def test_replace_identical_item_to_empty_gallery(tmp_path: Path):
    gallery = Gallery(json_file_path=str(tmp_path / "test.json"))

    identical_matrix = generate_realistic_embedding()

    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = identical_matrix.tolist()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [],  # No classifications
    ]

    gallery.update([mock_detection])
    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.SUCCESS
    )

    # Mock detection with dissimilar embedding
    mock_detection2 = MagicMock()
    mock_matrix2 = MagicMock()
    mock_matrix2.get_data.return_value = identical_matrix.tolist()
    mock_unique_id2 = MagicMock()
    mock_unique_id2.get_id.return_value = 12

    # Set up mock behavior
    mock_detection2.get_objects_typed.side_effect = [
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [],  # No classifications
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [],  # No classifications
    ]

    gallery.update([mock_detection2])
    assert (
        gallery.append_new_item(name="Max", append=True) == GalleryAppendStatus.SUCCESS
    )

    # Assertions
    calls = mock_detection.add_object.call_args_list
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 0
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert gallery.tracking_id_to_global_id[11] == 0

    calls = mock_detection2.add_object.call_args_list
    assert len(calls) == 3
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 0
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert isinstance(calls[1][0][0], hailo.HailoClassification)
    assert calls[1][0][0].get_label() == "Stefanos"
    assert gallery.tracking_id_to_global_id[12] == 0

    # Check if the file was created and written to
    assert os.path.exists(str(tmp_path / "test.json"))
    with open(str(tmp_path / "test.json"), "r") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["FaceRecognition"]["Name"] == "Max"
    assert len(data[0]["FaceRecognition"]["Embeddings"]) == 2
    assert (
        data[0]["FaceRecognition"]["Embeddings"][0]["HailoMatrix"]["data"]
        == identical_matrix.tolist()
    )
    assert (
        data[0]["FaceRecognition"]["Embeddings"][1]["HailoMatrix"]["data"]
        == identical_matrix.tolist()
    )


def test_replace_identical_item_with_the_same_name_to_empty_gallery(tmp_path: Path):
    gallery = Gallery(json_file_path=str(tmp_path / "test.json"))

    identical_matrix = generate_realistic_embedding()

    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = identical_matrix.tolist()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [],  # No classifications
    ]

    gallery.update([mock_detection])
    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.SUCCESS
    )

    # Mock detection with dissimilar embedding
    mock_detection2 = MagicMock()
    mock_matrix2 = MagicMock()
    mock_matrix2.get_data.return_value = identical_matrix.tolist()
    mock_unique_id2 = MagicMock()
    mock_unique_id2.get_id.return_value = 12

    # Set up mock behavior
    mock_detection2.get_objects_typed.side_effect = [
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [],  # No classifications
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [],  # No classifications
    ]

    gallery.update([mock_detection2])
    assert (
        gallery.append_new_item(name="Stefanos", append=True)
        == GalleryAppendStatus.SUCCESS
    )

    # Assertions
    calls = mock_detection.add_object.call_args_list
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 0
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert gallery.tracking_id_to_global_id[11] == 0

    calls = mock_detection2.add_object.call_args_list
    assert len(calls) == 3
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 0
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert isinstance(calls[1][0][0], hailo.HailoClassification)
    assert calls[1][0][0].get_label() == "Stefanos"
    assert gallery.tracking_id_to_global_id[12] == 0

    # Check if the file was created and written to
    assert os.path.exists(str(tmp_path / "test.json"))
    with open(str(tmp_path / "test.json"), "r") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["FaceRecognition"]["Name"] == "Stefanos"
    assert len(data[0]["FaceRecognition"]["Embeddings"]) == 2
    assert (
        data[0]["FaceRecognition"]["Embeddings"][0]["HailoMatrix"]["data"]
        == identical_matrix.tolist()
    )
    assert (
        data[0]["FaceRecognition"]["Embeddings"][1]["HailoMatrix"]["data"]
        == identical_matrix.tolist()
    )


def test_add_identical_item_to_empty_gallery_item_exists(tmp_path: Path):
    gallery = Gallery(json_file_path=str(tmp_path / "test.json"))

    identical_matrix = generate_realistic_embedding()

    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = identical_matrix.tolist()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [],  # No classifications
    ]

    gallery.update([mock_detection])
    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.SUCCESS
    )

    # Mock detection with dissimilar embedding
    mock_detection2 = MagicMock()
    mock_matrix2 = MagicMock()
    mock_matrix2.get_data.return_value = identical_matrix.tolist()
    mock_unique_id2 = MagicMock()
    mock_unique_id2.get_id.return_value = 12

    # Set up mock behavior
    mock_detection2.get_objects_typed.side_effect = [
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [],  # No classifications
        [mock_unique_id2],
        [mock_matrix2],
        [mock_unique_id2],
        [],  # No classifications
    ]

    gallery.update([mock_detection2])
    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.ITEM_ALREADY_EXISTS
    )


def test_add_item_to_empty_gallery_no_faces_found(tmp_path: Path):
    gallery = Gallery(json_file_path=str(tmp_path / "test.json"))

    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.NO_FACES_FOUND
    )


def test_add_item_to_empty_gallery_multiple_faces_found(tmp_path: Path):
    gallery = Gallery(json_file_path=str(tmp_path / "test.json"))

    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = generate_realistic_embedding()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [],  # No classifications
    ]

    gallery.update([mock_detection, mock_detection])

    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.MULTIPLE_FACES_FOUND
    )


def test_delete_one_item(tmp_path: Path):
    gallery = Gallery(json_file_path=str(tmp_path / "test.json"))

    identical_matrix = generate_realistic_embedding()

    # Mock detection with dissimilar embedding
    mock_detection = MagicMock()
    mock_matrix = MagicMock()
    mock_matrix.get_data.return_value = identical_matrix.tolist()
    mock_unique_id = MagicMock()
    mock_unique_id.get_id.return_value = 11

    # Set up mock behavior
    mock_detection.get_objects_typed.side_effect = [
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [mock_matrix],
        [mock_unique_id],
        [],  # No classifications
    ]

    gallery.update([mock_detection])
    assert len(gallery.tracking_id_to_global_id) == 0
    assert (
        gallery.append_new_item(name="Stefanos", append=False)
        == GalleryAppendStatus.SUCCESS
    )
    assert len(gallery.tracking_id_to_global_id) == 1
    assert (
        gallery.delete_item_by_name("Stefanos")
        == GalleryDeletionStatus.SUCCESS
    )
    # Assertions
    calls = mock_detection.add_object.call_args_list
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 0
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert len(gallery.tracking_id_to_global_id) == 0

    # Check if the file was created and written to
    assert os.path.exists(str(tmp_path / "test.json"))
    with open(str(tmp_path / "test.json"), "r") as f:
        data = json.load(f)
    assert len(data) == 0

def test_delete_one_item_not_found(gallery_and_json):
    gallery, test_json_path = gallery_and_json

    assert (
        gallery.delete_item_by_name("Stefanos")
        == GalleryDeletionStatus.NOT_FOUND
    )

    # Check if the file was created and written to
    assert os.path.exists(str(test_json_path))
    with open(str(test_json_path), "r") as f:
        data = json.load(f)
    assert len(data) == 2
