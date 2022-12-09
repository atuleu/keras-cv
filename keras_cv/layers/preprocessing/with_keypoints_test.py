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

TEST_CONFIGURATIONS = [
    ("AutoContrast", preprocessing.AutoContrast, {"value_range": (0, 255)}, True),
    ("ChannelShuffle", preprocessing.ChannelShuffle, {}, True),
    ("Equalization", preprocessing.Equalization, {"value_range": (0, 255)}, True),
    ("Grayscale", preprocessing.Grayscale, {}, True),
    (
        "Posterization",
        preprocessing.Posterization,
        {"bits": 3, "value_range": (0, 255)},
        True,
    ),
    (
        "RandomColorDegeneration",
        preprocessing.RandomColorDegeneration,
        {"factor": 0.5},
        True,
    ),
    (
        "RandomHue",
        preprocessing.RandomHue,
        {"factor": 0.5, "value_range": (0, 255)},
        True,
    ),
    (
        "RandomChannelShift",
        preprocessing.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.5},
        True,
    ),
    (
        "RandomColorJitter",
        preprocessing.RandomColorJitter,
        {
            "value_range": (0, 255),
            "brightness_factor": (-0.2, 0.5),
            "contrast_factor": (0.5, 0.9),
            "saturation_factor": (0.5, 0.9),
            "hue_factor": (0.5, 0.9),
            "seed": 1,
        },
        True,
    ),
    (
        "RandomGaussianBlur",
        preprocessing.RandomGaussianBlur,
        {"kernel_size": 3, "factor": (0.0, 3.0)},
        True,
    ),
    ("RandomJpegQuality", preprocessing.RandomJpegQuality, {"factor": (75, 100)}, True),
    ("RandomSaturation", preprocessing.RandomSaturation, {"factor": 0.5}, True),
    (
        "RandomSharpness",
        preprocessing.RandomSharpness,
        {"factor": 0.5, "value_range": (0, 255)},
        True,
    ),
    ("Solarization", preprocessing.Solarization, {"value_range": (0, 255)}, True),
]


class WithKeypointsTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_with_keypoints_single_image(self, layer_cls, init_args, noop):
        layer = layer_cls(**init_args)
        image = tf.random.uniform(
            shape=(12, 12, 3), minval=0, maxval=1, dtype=tf.float32
        )
        keypoints = tf.random.uniform(shape=(13, 8, 2), dtype=tf.float32)
        keypoints_mask = tf.random.uniform(shape=(13, 8), dtype=tf.float32) > 0.5
        keypoints = tf.ragged.boolean_mask(keypoints, keypoints_mask)

        inputs = {"images": image, "keypoints": keypoints}
        outputs = layer(inputs)
        self.assertIn("keypoints", outputs)
        if noop:
            self.assertAllClose(inputs["keypoints"], outputs["keypoints"])
