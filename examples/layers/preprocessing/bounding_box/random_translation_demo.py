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
"""
   random_trasnslation_demo.py shows how to use the RandomTranslation preprocessing layer
   for object detection.
"""
import demo_utils

from keras_cv.layers import preprocessing

IMG_SIZE = (256, 256)
BATCH_SIZE = 9


def main():
    inputs = demo_utils.load_voc_dataset(bounding_box_format="rel_xyxy")
    random_translation_layer = preprocessing.RandomTranslation(
        width_factor=0.5,
        height_factor=0.5,
        bounding_box_format="rel_xyxy",
        fill_mode="constant",
        fill_value=127.0,
    )
    input = next(iter(inputs.take(1)))
    outputs = random_translation_layer(input)
    demo_utils.visualize_data(outputs, bounding_box_format="rel_xyxy")


if __name__ == "__main__":
    main()
