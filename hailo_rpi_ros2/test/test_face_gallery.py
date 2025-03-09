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
    global_id = gallery.create_new_global_id()
    matrix = np.array([1, 2, 3])
    gallery.add_embedding(global_id, matrix)
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

    global_id1 = gallery.create_new_global_id()
    gallery.add_embedding(global_id1, matrix1)

    global_id2 = gallery.create_new_global_id()
    gallery.add_embedding(global_id2, matrix2)

    global_id3 = gallery.create_new_global_id()
    gallery.add_embedding(global_id3, matrix3)

    closest_id, distance = gallery.get_closest_global_id(matrix1)

    assert closest_id == global_id1

    # Add an extra assertion to check if a realistic distance is returned.
    # The distance should be small when comparing the same face.
    assert distance < 0.2

    # Test with a matrix very similar to matrix 2.
    matrix4 = matrix2 + np.random.normal(loc=0.0, scale=0.01, size=512)
    closest_id2, distance2 = gallery.get_closest_global_id(matrix4)
    assert closest_id2 == global_id2
    assert distance2 < 0.1
