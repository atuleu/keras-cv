# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from keras_cv.utils import preprocessing
from absl.testing import parameterized


class MockRandomGenerator:
    def __init__(self, value):
        self.value = value

    def random_uniform(self, shape, minval, maxval, dtype=None):
        del minval, maxval
        return tf.constant(self.value, dtype=dtype)


class PreprocessingTestCase(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()

    def test_transform_to_standard_range_neg_one_range(self):
        x = tf.constant([-1, 0, 1])
        x = preprocessing.transform_value_range(
            x, original_range=[-1, 1], target_range=[0, 255]
        )
        self.assertAllClose(x, [0.0, 127.5, 255.0])

    def test_transform_to_same_range(self):
        x = tf.constant([-1, 0, 1])
        x = preprocessing.transform_value_range(
            x, original_range=[0, 255], target_range=[0, 255]
        )
        self.assertAllClose(x, [-1, 0, 1])

    def test_transform_to_standard_range(self):
        x = tf.constant([8 / 255, 9 / 255, 255 / 255])
        x = preprocessing.transform_value_range(
            x, original_range=[0, 1], target_range=[0, 255]
        )
        self.assertAllClose(x, [8.0, 9.0, 255.0])

    def test_transform_to_value_range(self):
        x = tf.constant([128.0, 255.0, 0.0])
        x = preprocessing.transform_value_range(
            x, original_range=[0, 255], target_range=[0, 1]
        )
        self.assertAllClose(x, [128 / 255, 1, 0])

    def test_random_inversion(self):
        generator = MockRandomGenerator(0.75)
        self.assertEqual(preprocessing.random_inversion(generator), -1.0)
        generator = MockRandomGenerator(0.25)
        self.assertEqual(preprocessing.random_inversion(generator), 1.0)

    @parameterized.named_parameters(
        ("dense", [1, 2], {"dtype": "float32"}, tf.constant([1, 2], tf.float32)),
        (
            "sparse",
            tf.SparseTensor([[1], [2]], [2, 3], [4]),
            {},
            tf.constant([0, 2, 3, 0]),
        ),
        (
            "ragged",
            tf.ragged.constant([[[1, 2], [3, 4]], [], [[1], [2], [3]]]),
            {"default_value": 5},
            tf.constant(
                [
                    [[1, 2], [3, 4], [5, 5]],
                    [[5, 5], [5, 5], [5, 5]],
                    [[1, 5], [2, 5], [3, 5]],
                ]
            ),
        ),
    )
    def test_ensure_dense_tensor(self, inputs, kwargs, expected):
        self.assertAllClose(
            preprocessing.ensure_dense_tensor(inputs, **kwargs), expected
        )
