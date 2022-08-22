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
from absl.testing import parameterized

from keras_cv.layers import preprocessing

# Imports from with_labels the list of augmentation that should perform a No-Op.
from keras_cv.layers.preprocessing.with_labels_test import TEST_CONFIGURATIONS

# List here the layers that are expected to modify bounding boxes with
# their specific parameters.
GEOMETRIC_TEST_CONFIGURATIONS = [
    (
        "RandomShear",
        preprocessing.RandomShear,
        {
            "x_factor": 0.3,
            "x_factor": 0.3,
            "bounding_box_format": "xyxy",
        },
    ),
]


class WithBoundingBoxesTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        *TEST_CONFIGURATIONS,
        *GEOMETRIC_TEST_CONFIGURATIONS,
    )
    def test_can_run_with_bounding_boxes(self, layer_cls, init_args):
        layer = layer_cls(**init_args)

        img = tf.random.uniform(
            shape=(3, 512, 512, 3), minval=0, maxval=1, dtype=tf.float32
        )
        bounding_boxes = tf.ones((3, 2, 4), dtype=tf.float32)

        inputs = {"images": img, "bounding_boxes": bounding_boxes}
        outputs = layer(inputs)
        self.assertTrue("bounding_boxes" in outputs)

    # this has to be a separate test case to exclude CutMix and MixUp
    @parameterized.named_parameters(
        *TEST_CONFIGURATIONS,
        *GEOMETRIC_TEST_CONFIGURATIONS,
    )
    def test_can_run_with_bouding_boxes_single_image(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        img = tf.random.uniform(
            shape=(512, 512, 3), minval=0, maxval=1, dtype=tf.float32
        )
        bounding_boxes = tf.ones((3, 4), dtype=tf.float32)
        inputs = {"images": img, "bounding_boxes": bounding_boxes}
        outputs = layer(inputs)
        self.assertTrue("bounding_boxes" in outputs)

    # CutMix needs labels data
    def test_cut_mix_keeps_bounding_box_data(self):
        layer = preprocessing.CutMix()
        img = tf.ones(shape=(3, 512, 512, 3), dtype=tf.float32)
        labels = tf.ones((3), dtype=tf.float32)
        bounding_boxes = tf.reshape(
            tf.range(3 * 2 * 4, dtype=tf.float32), shape=(3, 2, 4)
        )
        inputs = {"images": img, "bounding_boxes": bounding_boxes, "labels": labels}
        outputs = layer(inputs)
        self.assertAllEqual(inputs["bounding_boxes"], outputs["bounding_boxes"])
