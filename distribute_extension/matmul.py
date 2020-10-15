import tensorflow as tf
from device_utils import limit_to_virtual_gpus


@tf.custom_gradient
def matmul_mp(M, P):
    """Partitioned Matrix Multiplication.

    The `_pm` suffix means the first matrix is partitioned, along the last
    dimension and the second matrix is mirrored.

    $$
    M P = M \begin{bmatrix}P_1 & P_2 & \cdots \end{bmatrix}
        = \begin{bmatrix}M P_1 & M P_2 & \cdots \end{bmatrix}
    $$

    where
        M is a mirrored matrix, and
        P is a partitioned matrix.

    If we view matrix M as of shape [batchsize, input_feature_dim] and matrix P
    as of shape [input_feature_dim, output_feature dim], then the computation
    will be reduced to different devices and the result will be spreaded. This
    let us do large matrix multiplication with lots of "small" compute unit,
    instead of being constrained to a huge computer (HPC).

    This is effectively a model parallelism case."""

    M = tf.stop_gradient(M)
    P = tf.stop_gradient(P)

    def grad_fn(grad):
        ctx = tf.distribute.get_replica_context()

        grad_M = tf.matmul(grad, tf.transpose(P))
        grad_M = ctx.all_reduce(tf.distribute.ReduceOp.SUM, grad_M)
        # tf.print(ctx.replica_id_in_sync_group, "grad_M", grad_M)

        grad_P = tf.matmul(tf.transpose(M), grad)

        return grad_M, grad_P

    return tf.matmul(M, P), grad_fn
