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
"""keypoints_demo.py shows how to use geometric preprocessing layer to
augment an image and associated keypoints.

Operates on the AFLWL20003D dataset.  In this script the face images
and their 2D keypoint data are loaded, then are passed through the
preprocessing layers applying random shear, translation or rotation to
them. Finally, they are shown using matplotlib. Augmented keypoints
falling outside the images boundaries are discarded.
"""
import demo_utils

from keras_cv.layers import preprocessing


def main():
    ds = demo_utils.load_AFLW2000_dataset(batch_size=9)
    layer = preprocessing.RandomTranslation(
        width_factor=0.6,
        height_factor=0.6,
        keypoint_format="xy",
        fill_mode="constant",
        fill_value=127.0,
    )
    ds = ds.map(layer)
    demo_utils.visualize_data(ds)


if __name__ == "__main__":
    main()
