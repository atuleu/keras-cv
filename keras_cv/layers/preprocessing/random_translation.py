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

import warnings

import tensorflow as tf

from keras_cv import bounding_box
from keras_cv import keypoint
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomTranslation(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly translates images during training.

    This layer will apply random translations to each image, filling empty space
    according to `fill_mode`.

    By default, random translations are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    translations at inference time, set `training` to True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Arguments:
      height_factor: a float represented as fraction of value, or a tuple of
        size 2 representing lower and upper bound for shifting vertically. A
        negative value means shifting image up, while a positive value means
        shifting image down. When represented as a single positive float, this
        value is used for both the upper and lower bound. For instance,
        `height_factor=(-0.2, 0.3)` results in an output shifted by a random
        amount in the range `[-20%, +30%]`.  `height_factor=0.2` results in an
        output height shifted by a random amount in the range `[-20%, +20%]`.
      width_factor: a float represented as fraction of value, or a tuple of size
        2 representing lower and upper bound for shifting horizontally. A
        negative value means shifting image left, while a positive value means
        shifting image right. When represented as a single positive float, this
        value is used for both the upper and lower bound. For instance,
        `width_factor=(-0.2, 0.3)` results in an output shifted left by 20%, and
        shifted right by 30%. `width_factor=0.2` results in an output height
        shifted left or right by 20%.
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
        - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
          reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
          filling all values beyond the edge with the same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
          wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
          the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode="constant"`.
      bounding_box_format: The format of bounding boxes of input dataset. Refer
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported bounding box formats.
      keypoint_format: The format of keypoints of input dataset. Refer
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/keypoint/converters.py
        for more details on supported keypoint formats.

    """

    def __init__(
        self,
        height_factor=None,
        width_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        bounding_box_format=None,
        keypoint_format=None,
        **kwargs,
    ):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.seed = seed
        if height_factor is not None:
            self.height_factor = preprocessing.parse_symmetric_factor(
                param=height_factor, max_value=1.0, param_name="height_factor"
            )
        else:
            self.height_factor = None
        if width_factor is not None:
            self.width_factor = preprocessing.parse_symmetric_factor(
                param=width_factor, max_value=1.0, param_name="width_factor"
            )
        else:
            self.width_factor = None

        if height_factor is None and width_factor is None:
            warnings.warn(
                "RandomTranslation received both `heigt_factor=None` and "
                "`width_factor=None`.  As a result, the layer will perform no "
                "augmentation."
            )

        self.fill_mode = preprocessing.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format
        self.keypoint_format = keypoint_format

    def get_random_transformation(self, image, **kwargs):
        image_size = tf.cast(tf.shape(image), self.compute_dtype)
        image_height, image_width = image_size[-3], image_size[-2]

        height_px = 0.0
        width_px = 0.0
        if self.height_factor:
            height_px = self.height_factor() * image_height
        if self.width_factor:
            width_px = self.width_factor() * image_width

        return tf.stack([width_px, height_px])

    def augment_image(self, image, transformation, **kwargs):
        image = preprocessing.ensure_tensor(image, self.compute_dtype)
        image = tf.expand_dims(image, axis=0)

        translations = preprocessing.get_translation_matrix(
            tf.expand_dims(transformation, axis=0)
        )

        image = preprocessing.transform(
            images=image,
            transforms=translations,
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        return tf.squeeze(image, axis=0)

    def augment_label(self, labels, transformation, **kwargs):
        return labels

    def augment_keypoints(self, keypoints, image, transformation, **kwargs):
        keypoint_format = kwargs.get("keypoint_format", self.keypoint_format)
        if keypoint_format is None:
            raise preprocessing.build_missing_keypoint_format_error("RandomTranslation")

        xy, rest = tf.split(keypoints, [2, tf.shape(keypoints)[-1] - 2], axis=-1)

        expanded_image = tf.expand_dims(image, axis=0)

        xy = keypoint.convert_format(
            tf.expand_dims(xy, axis=0),
            source=keypoint_format,
            target="xy",
            images=expanded_image,
        )

        out_xy = xy + transformation

        out = keypoint.convert_format(
            out_xy, source="xy", target=keypoint_format, images=expanded_image
        )

        keypoints_mask = keypoint.build_inside_mask(
            out, expanded_image, keypoint_format=keypoint_format
        )
        keypoints = tf.concat([tf.squeeze(out, axis=0), rest], axis=-1)
        keypoints_mask = tf.squeeze(keypoints_mask, axis=0)

        return keypoints, keypoints_mask

    def augment_bounding_boxes(self, bounding_boxes, image, transformation, **kwargs):
        if self.bounding_box_format is None:
            raise preprocessing.build_missing_bounding_box_format_error(
                "RandomTranslation"
            )

        return bounding_box.transform_from_corners_fn(
            bounding_boxes,
            transform_corners_fn=lambda corners: self.augment_keypoints(
                corners,
                transformation=transformation,
                image=image,
                keypoint_format="xy",
            )[0],
            images=image,
            bounding_box_format=self.bounding_box_format,
            clip_boxes=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height_factor": self.height_factor,
                "width_factor": self.width_factor,
                "fill_mode": self.fill_mode,
                "interpolation": self.interpolation,
                "fill_value": self.fill_value,
                "bounding_box_format": self.bounding_box_format,
                "keypoint_format": self.keypoint_format,
                "seed": self.seed,
            }
        )
        return config
