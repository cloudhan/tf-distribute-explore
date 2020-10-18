import pytest
import tensorflow as tf

import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from device_utils import limit_to_virtual_gpus
from softmax_cross_entropy import softmax_cross_entropy_with_logits


@pytest.mark.parametrize("batchsize, num_class_pr", zip([23, 32, 64, 128, 256], [4, 8, 25, 250, 25000]))
def test_softmax_cross_entropy_with_logits(batchsize, num_class_pr):
    gpus = limit_to_virtual_gpus()
    dp = tf.distribute.MirroredStrategy(devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    num_replica = dp.num_replicas_in_sync

    coeff = tf.random.uniform(shape=(batchsize,)) + 1

    logits = tf.range(batchsize * num_class_pr * num_replica, dtype=tf.float32)
    logits = tf.reshape(logits, shape=(batchsize, -1))
    labels = tf.math.mod(tf.range(batchsize), num_class_pr * num_replica)

    with tf.GradientTape() as tape:
        tape.watch(logits)
        ref = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = coeff * ref
    (ref_grad,) = tape.gradient(loss, [logits])

    logits = tf.reshape(logits, shape=(batchsize, num_replica, num_class_pr))
    logits = dp.experimental_distribute_values_from_function(lambda ctx: logits[:, ctx.replica_id_in_sync_group, :])
    coeff = dp.experimental_distribute_values_from_function(lambda ctx: coeff)

    with dp.scope():
        labels = tf.Variable(labels)

    # NOTE: softmax_cross_entropy_with_logits is not a tf.function, the bug will
    # not be triggered. for simplicity here, we will not wrap the whole
    # computation in a tf.function.
    def forward_backward(logits, labels, coeff):  # def step(...):
        with tf.GradientTape() as tape:
            tape.watch(logits)
            my = softmax_cross_entropy_with_logits(logits, labels)
            loss = coeff * my

        (grad,) = tape.gradient(loss, [logits])
        return my, grad

    my, my_grad = dp.run(forward_backward, args=(logits, labels, coeff))
    my_grad = tf.concat(my_grad.values, axis=1)

    assert ref.shape == my.values[0].shape
    assert ref_grad.shape == my_grad.shape

    assert tf.reduce_max(tf.abs(ref - my.values[0])) < 1e-6
    assert tf.reduce_max(tf.abs(ref_grad - my_grad)) < 1e-6


if __name__ == "__main__":
    test_softmax_cross_entropy_with_logits(32, 4)
