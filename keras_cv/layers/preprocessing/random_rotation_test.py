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
import numpy as np
import tensorflow as tf

from keras_cv.layers.preprocessing.random_rotation import RandomRotation


class RandomRotationTest(tf.test.TestCase):
    def test_random_rotation_output_shapes(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = RandomRotation(0.5)
        actual_output = layer(input_images, training=True)
        self.assertEqual(expected_output.shape, actual_output.shape)

    def test_random_rotation_inference(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = RandomRotation(0.5)
        actual_output = layer(input_images, training=False)
        self.assertAllClose(expected_output, actual_output)

    def test_config_with_custom_name(self):
        layer = RandomRotation(0.5, name="image_preproc")
        config = layer.get_config()
        layer_reconstructed = RandomRotation.from_config(config)
        self.assertEqual(layer_reconstructed.name, layer.name)

    def test_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(np.float32)
        # 180 rotation.
        layer = RandomRotation(factor=(0.5, 0.5))
        output_image = layer(input_image)
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        ).astype(np.float32)
        expected_output = np.reshape(expected_output, (5, 5, 1))
        self.assertAllClose(expected_output, output_image)

    def test_augment_bbox_dict_input(self):
        input_image = np.random.random((512, 512, 3)).astype(np.float32)
        # Makes sure it supports additionnal data attached to bboxes,
        # like classes and confidences
        bboxes = tf.convert_to_tensor(
            [[200, 200, 400, 400, 3.0], [100, 100, 300, 300, 42.0]]
        )
        input = {"images": input_image, "bounding_boxes": bboxes}
        # 180 rotation.
        layer = RandomRotation(factor=(0.5, 0.5), bounding_box_format="xyxy")
        output_bbox = layer(input)
        expected_output = np.asarray(
            [[112.0, 112.0, 312.0, 312.0, 3.0], [212.0, 212.0, 412.0, 412.0, 42.0]],
        )
        expected_output = np.reshape(expected_output, (2, 5))
        self.assertAllClose(expected_output, output_bbox["bounding_boxes"])

    def test_augment_keypoints_dict_input(self):
        input_image = np.random.random((512, 512, 3)).astype(np.float32)
        # Makes sure it supports additionnal data attached to bboxes,
        # like classes and confidences
        keypoints = tf.convert_to_tensor(
            [
                [[200.0, 200.0], [400.0, 400.0]],
                [[100.0, 100.0], [300.0, 300.0]],
            ]
        )
        input = {"images": input_image, "keypoints": keypoints}
        # 180 rotation.
        layer = RandomRotation(factor=(0.5, 0.5), keypoint_format="xy")
        output_bbox = layer(input)
        expected_output = np.asarray(
            [
                [[312.0, 312.0], [112.0, 112.0]],
                [[412.0, 412.0], [212.0, 212.0]],
            ]
        )
        self.assertAllClose(expected_output, output_bbox["keypoints"])

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomRotation(0.5)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = RandomRotation(0.5, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")
