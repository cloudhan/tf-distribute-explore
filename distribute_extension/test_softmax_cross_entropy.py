import pytest
import tensorflow as tf
from .device_utils import limit_to_virtual_gpus
from .softmax_cross_entropy import softmax_cross_entropy_with_logits


@pytest.mark.parametrize("batchsize, num_class_pr", zip([23, 32, 64, 128], [4, 8, 25, 250]))
def test_softmax_cross_entropy_with_logits(batchsize, num_class_pr):
    gpus = limit_to_virtual_gpus()
    dp = tf.distribute.MirroredStrategy(devices=gpus)
    num_replica = dp.num_replicas_in_sync

    logits = tf.range(batchsize * num_class_pr * num_replica, dtype=tf.float32)
    logits_partitioned = tf.reshape(logits, shape=(batchsize, num_replica, num_class_pr))
    logits_partitioned = dp.experimental_distribute_values_from_function(
        lambda ctx: logits_partitioned[:, ctx.replica_id_in_sync_group, :]
    )
    logits = tf.reshape(logits, shape=(batchsize, -1))

    labels = tf.math.mod(tf.range(batchsize), num_class_pr * num_replica)

    with dp.scope():
        labels_mirrored = tf.Variable(labels)

    with tf.GradientTape() as tape:
        tape.watch(logits)
        ref = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    (ref_grad,) = tape.gradient(ref, [logits])

    # NOTE: softmax_cross_entropy_with_logits is not a tf.function, the bug will
    # not be triggered. for simplicity here, we will not wrap the whole
    # computation in a tf.function.
    def forward_backward(logits, labels):  # def step(...):
        with tf.GradientTape() as tape:
            tape.watch(logits)
            ret = softmax_cross_entropy_with_logits(logits, labels)

        assert tf.distribute.get_replica_context() is not None
        (grad,) = tape.gradient(ret, [logits])
        return ret, grad

    my, my_grad = dp.run(forward_backward, args=(logits_partitioned, labels_mirrored))

    forward_diff = ref - my.values[0]
    assert tf.reduce_max(tf.abs(forward_diff)) < 1e-7

    my_grad = tf.concat(my_grad.values, axis=1)
    backward_diff = ref_grad - my_grad
    assert tf.reduce_max(tf.abs(backward_diff)) < 1e-7
