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


def mark_out_of_image_as_sentinel(
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
    as_xy = convert_format(
        keypoints,
        images=images,
        image_shape=image_shape,
        source=keypoint_format,
        target="xy",
    )
    image_height, image_width = _image_shape(images, image_shape, keypoints)
    outside = tf.math.logical_or(
        tf.math.logical_or(as_xy[..., 0] < 0, as_xy[..., 0] >= image_width),
        tf.math.logical_or(as_xy[..., 1] < 0, as_xy[..., 1] >= image_height),
    )
    xy, val, rest = tf.split(keypoints, [2, 1, keypoints.shape[-1] - 3], axis=-1)
    val = tf.where(outside[..., None], tf.cast(sentinel_value, val.dtype), val)
    return tf.concat([xy, val, rest], axis=-1)


def filter_sentinels(keypoints, sentinel_value=-1):
    isragged = isinstance(keypoints, tf.RaggedTensor)
    if isragged:
        keypoints = keypoints.to_tensor(default_value=sentinel_value)

    mask = keypoints[..., 2] != sentinel_value
    return tf.ragged.boolean_mask(keypoints, mask)
