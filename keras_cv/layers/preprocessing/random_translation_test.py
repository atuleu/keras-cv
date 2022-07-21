import tensorflow as tf
from absl.testing import parameterized

from keras_cv.layers.preprocessing import RandomTranslation

images = tf.ones((2, 50, 50, 1))
bounding_boxes = tf.constant(
    [
        [[1, 2, 12, 13, 1], [2, 3, 22, 23, 2]],
        [[3, 4, 33, 34, 3], [4, 5, 44, 45, 4]],
    ],
    tf.float32,
)
keypoints = tf.constant(
    [
        [[[1, 2, 1], [12, 13, 2]], [[2, 3, 3], [22, 23, 4]]],
        [[[3, 4, 5], [33, 34, 6]], [[4, 5, 7], [44, 45, 8]]],
    ],
    tf.float32,
)

translated_bounding_boxes = {
    (0.0, 0.5): tf.minimum(
        bounding_boxes + tf.constant([25, 0, 25, 0, 0], tf.float32), 50.0
    ),
    (0.5, 0.0): tf.minimum(
        bounding_boxes + tf.constant([0, 25, 0, 25, 0], tf.float32), 50.0
    ),
}

translated_keypoints = {
    (0.0, 0.5): (
        keypoints + tf.constant([25, 0, 0], tf.float32),
        tf.constant([[[True, True], [True, True]], [[True, False], [True, False]]]),
    ),
    (0.5, 0.0): (
        keypoints + tf.constant([0, 25, 0], tf.float32),
        tf.constant([[[True, True], [True, True]], [[True, False], [True, False]]]),
    ),
}


class RandomTranslationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "width",
            0.0,
            0.5,
            tf.constant(
                [
                    [1, 0, 0, 1],
                    [5, 4, 4, 5],
                    [9, 8, 8, 9],
                    [13, 12, 12, 13],
                ]
            ),
        ),
        (
            "height",
            0.5,
            0.0,
            tf.constant(
                [
                    [4, 5, 6, 7],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ]
            ),
        ),
    )
    def test_augment_image_width(self, height_factor, width_factor, expected_output):
        image = tf.reshape(tf.range(16), (4, 4, 1))
        expected_output = tf.expand_dims(expected_output, axis=-1)
        layer = RandomTranslation(
            height_factor=(height_factor, height_factor),
            width_factor=(width_factor, width_factor),
        )
        self.assertAllClose(layer(image), expected_output)

    @parameterized.named_parameters(
        ("width", 0.0, 0.5),
        ("height", 0.5, 0.0),
    )
    def test_with_bbox_dict_output(self, height_factor, width_factor):
        expected_bbox = translated_bounding_boxes[(height_factor, width_factor)]
        layer = RandomTranslation(
            height_factor=(height_factor, height_factor),
            width_factor=(width_factor, width_factor),
            bounding_box_format="xyxy",
        )
        output = layer({"images": images, "bounding_boxes": bounding_boxes})
        self.assertAllClose(output["bounding_boxes"], expected_bbox)

    @parameterized.named_parameters(
        ("width", 0.0, 0.5),
        ("height", 0.5, 0.0),
    )
    def test_with_keypoints_dict_output(self, height_factor, width_factor):
        expected_keypoints, expected_keypoints_mask = translated_keypoints[
            (height_factor, width_factor)
        ]
        layer = RandomTranslation(
            height_factor=(height_factor, height_factor),
            width_factor=(width_factor, width_factor),
            keypoint_format="xy",
        )
        output = layer({"images": images, "keypoints": keypoints})
        self.assertAllClose(output["keypoints"], expected_keypoints)
        self.assertAllClose(output["keypoints_mask"], expected_keypoints_mask)
