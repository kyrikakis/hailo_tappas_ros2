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

import pytest
import numpy as np
from hailo_rpi_ros2.face_gallery import Gallery, gallery_one_dim_dot_product, gallery_get_xtensor
from unittest.mock import MagicMock, call
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


def generate_realistic_embedding(size=512):
    return np.random.normal(loc=0.0, scale=0.1, size=size)


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

def test_new_embedding_to_global_id():
    # Create a test json file
    matrix1 = generate_realistic_embedding()
    test_json_path = "test_gallery.json"
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
                            "data": matrix1.tolist()
                        }
                    }
                ]
            }
        }
    ]
    with open(test_json_path, 'w') as f:
        json.dump(test_data, f)

    gallery = Gallery()
    gallery.load_local_gallery_from_json(test_json_path) #Load the test json

    matrix2 = generate_realistic_embedding()
    matrix3 = matrix1 + np.random.normal(loc=0.0, scale=0.1, size=512)
    matrix4 = generate_realistic_embedding() + 1

    mock_detection1 = MagicMock()
    mock_detection2 = MagicMock()
    mock_detection3 = MagicMock()
    mock_detection4 = MagicMock()

    mock_matrix1 = MagicMock()
    mock_matrix1.get_data.return_value = matrix1
    mock_matrix2 = MagicMock()
    mock_matrix2.get_data.return_value = matrix2
    mock_matrix3 = MagicMock()
    mock_matrix3.get_data.return_value = matrix3
    mock_matrix4 = MagicMock()
    mock_matrix4.get_data.return_value = matrix4

    mock_unique_id1 = MagicMock()
    mock_unique_id1.get_id.return_value = 1
    mock_unique_id2 = MagicMock()
    mock_unique_id2.get_id.return_value = 2
    mock_unique_id3 = MagicMock()
    mock_unique_id3.get_id.return_value = 3
    mock_unique_id4 = MagicMock()
    mock_unique_id4.get_id.return_value = 4

    # Case 1: Track ID already exists (and local embeddings loaded)
    gallery.tracking_id_to_global_id[1] = 1
    mock_detection1.get_objects_typed.side_effect = [[mock_matrix1], [mock_unique_id1], []]
    gallery._new_embedding_to_global_id(matrix3, mock_detection1, 1)

     # Assert that HailoUniqueID and HailoClassification were added
    calls = mock_detection1.add_object.call_args_list
    assert len(calls) == 2
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 1
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert isinstance(calls[1][0][0], hailo.HailoClassification)
    assert calls[1][0][0].get_label() == "Alon"


    # Case 2: No embedding in detection
    mock_detection2.get_objects_typed.return_value = []
    gallery._new_embedding_to_global_id(None, mock_detection2, 2)
    mock_detection2.add_object.assert_not_called()

    # Case 3: Similar embedding found (and local embeddings loaded)
    mock_detection3.get_objects_typed.side_effect = [[mock_matrix3], [mock_unique_id3], []]
    gallery._new_embedding_to_global_id(matrix3, mock_detection3, 3)
    calls = mock_detection3.add_object.call_args_list
    assert len(calls) == 2
    assert isinstance(calls[0][0][0], hailo.HailoUniqueID)
    assert calls[0][0][0].get_id() == 1
    assert calls[0][0][0].get_mode() == hailo.GLOBAL_ID
    assert isinstance(calls[1][0][0], hailo.HailoClassification)
    assert calls[1][0][0].get_label() == "Alon"
    assert gallery.tracking_id_to_global_id[3] == 1

    # Case 4: Dissimilar embedding found
    mock_detection4.get_objects_typed.side_effect = [[mock_matrix4], [mock_unique_id4], []]
    gallery._new_embedding_to_global_id(matrix4, mock_detection4, 4)
    mock_detection4.add_object.assert_not_called()
    assert 4 not in gallery.tracking_id_to_global_id

    os.remove(test_json_path) #Clean up the test json file.