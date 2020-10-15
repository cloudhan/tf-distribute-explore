import pytest
import tensorflow as tf

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from device_utils import limit_to_virtual_gpus
from matmul import matmul_mp

mnk = [
    (3, 5, 7),
    (3, 5, 11),
    (3, 11, 5),
    (3, 29, 5),
    (29, 11, 5),
]

@pytest.mark.parametrize("m, n, k", mnk)
def test_matmul_mp(m, n, k):
    gpus = limit_to_virtual_gpus()
    dp = tf.distribute.MirroredStrategy(devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    num_replica = dp.num_replicas_in_sync

    K = k * num_replica

    coeff = tf.random.uniform(shape=(m, K))
    # NOTE: matrix multiplication involves summation, will cause numerical
    #  issue if it contains large elements.
    M = tf.random.uniform(shape=(m, n)) * 2 - 1
    P = tf.random.uniform(shape=(n, K)) * 2 - 1

    with tf.GradientTape() as tape:
        tape.watch(M)
        tape.watch(P)
        ref = tf.matmul(M, P)
        loss = tf.reduce_sum(coeff * ref * ref)
        tf.print("loss", loss)
    ref_M, ref_P = tape.gradient(loss, [M, P])

    with dp.scope():
        M = tf.identity(M)

    coeff = dp.experimental_distribute_values_from_function(lambda ctx: tf.slice(coeff, (0, k*ctx.replica_id_in_sync_group), (m, k)))
    P = dp.experimental_distribute_values_from_function(lambda ctx: tf.slice(P, (0, k*ctx.replica_id_in_sync_group), (n, k)))

    def forward_backward(M, P, coeff):
        ctx = tf.distribute.get_replica_context()
        with tf.GradientTape() as tape:
            tape.watch([M, P])
            forward = matmul_mp(M, P)
            # NOTE: just DON'T do ctx.all_reduce(...), just DON'T!
            loss = tf.reduce_sum(coeff * forward * forward)
        backward = tape.gradient(loss, [M, P])
        return forward, backward

    my, (my_M, my_P) = dp.run(forward_backward, args=(M, P, coeff))
    if isinstance(my, tf.distribute.DistributedValues):
        my = tf.concat(my.values, axis=-1)
        my_M = my_M.values[0]
        my_P = tf.concat(my_P.values, axis=-1)

    assert tf.reduce_max(tf.abs(ref - my)) < 5e-6
    assert tf.reduce_max(tf.abs(ref_M - my_M)) < 5e-6
    assert tf.reduce_max(tf.abs(ref_P - my_P)) < 5e-6


if __name__ == "__main__":
    test_matmul_mp(5, 7, 11)
