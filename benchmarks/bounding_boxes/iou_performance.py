import tensorflow as tf
import functools

from keras_cv import bounding_box


def create_random_data(batch_size=64, num_anchors=10_000, num_boxes=20):
    anchors = tf.random.uniform(shape=(batch_size, num_anchors, 4),
                                minval=0.0,
                                maxval=512,
                                dtype=tf.float32)
    boxes = tf.random.uniform(shape=(batch_size, num_boxes, 4),
                              minval=0.0,
                              maxval=512,
                              dtype=tf.float32)
    return anchors, boxes


def benchmark_iou(f, **kwargs):
    with tf.Graph().as_default() as graph:
        anchors, boxes = create_random_data(**kwargs)
        output = f(anchors, boxes, bounding_box_format="center_xyWH")
        with tf.compat.v1.Session(graph=graph) as sess:
            bm = tf.test.Benchmark()
            bm_result = bm.run_op_benchmark(sess, output)
    return bm_result


print("Vectorized=False")
benchmark_iou(functools.partial(bounding_box.compute_iou, vectorized=False))
print("Vectorized=True")
benchmark_iou(functools.partial(bounding_box.compute_iou, vectorized=True))
