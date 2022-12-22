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
"""Utility functions for keypoint transformation."""
import tensorflow as tf

H_AXIS = -3
W_AXIS = -2


from keras_cv.bounding_box.converters import _image_shape
from keras_cv.keypoint.converters import convert_format


def _inside_of_image_mask(
    keypoints, images=None, image_shape=None, keypoint_format=None
):
    if keypoint_format.startswith("rel_"):
        as_xy = keypoints
        image_height, image_width = (1.0, 1.0)
    else:
        as_xy = convert_format(
            keypoints,
            images=images,
            image_shape=image_shape,
            source=keypoint_format,
            target="xy",
        )
        image_height, image_width = _image_shape(images, image_shape, keypoints)
    inside = tf.math.logical_and(
        tf.math.logical_and(as_xy[..., 0] >= 0, as_xy[..., 0] < image_width),
        tf.math.logical_and(as_xy[..., 1] >= 0, as_xy[..., 1] < image_height),
    )
    return inside


def filter_out_of_image(
    keypoints, images=None, image_shape=None, keypoint_format=None, sentinel_value=-1
):
    """Discards keypoints if falling outside of the image.

    Args:
      keypoints: a, possibly ragged, 2D (ungrouped), 3D (grouped)
        keypoint data in the 'xy' format.
      image: a 3D tensor in the HWC format.

    Returns:
      tf.RaggedTensor: a 2D or 3D ragged tensor with at least one
        ragged rank containing only keypoint in the image.
    """
    inside = _inside_of_image_mask(keypoints, images, image_shape, keypoint_format)
    return tf.ragged.boolean_mask(keypoints, inside)


def mark_out_of_image_as_sentinel(
    keypoints, images=None, image_shape=None, keypoint_format=None, sentinel_value=-1
):
    inside = _inside_of_image_mask(keypoints, images, image_shape, keypoint_format)
    if keypoints.shape[-1] == 2:
        xy, cls, rest = tf.split(keypoints, [2, 0, 0], axis=-1)
    else:
        xy, cls, rest = tf.split(keypoints, [2, 1, keypoints.shape[-1] - 3], axis=-1)
    cls = tf.where(inside[..., None], cls, tf.cast(sentinel_value, cls.dtype))
    return tf.concat([xy, cls, rest], axis=-1)


def filter_sentinels(keypoints, sentinel_value=-1):
    isragged = isinstance(keypoints, tf.RaggedTensor)
    if isragged:
        keypoints = keypoints.to_tensor(default_value=sentinel_value)

    mask = keypoints[..., 2] != sentinel_value
    return tf.ragged.boolean_mask(keypoints, mask)
