import tensorflow as tf


@tf.custom_gradient
def softmax_cross_entropy_with_logits(logits, labels):
    """An Operator computes softmax loss partitioned on multiple devices.
    Assuming they are equally partitioned.

    Args:
        logits: PerReplica of Tensor (a.k.a, partitioned), e.g., If we have
            a PerReplica Value (with 4 replica) of Tensors be of shape
            128 * 250_000. Then batchsize is 128, and we are doing 1 million
            classes classification (250000 * 4 replicas), with each replica
            handles only 250000 classes. This is called model parallelism.
        labels: MirroredVariable of Tensor

    Note:
        To numerically stably compute $softmax(x) = e^{x_i} / \sum_j e^{x_j}$,
        simply subtract $$\max_j x_j$$ (abbr. max). To numerically stably
        compute cross_entropy(softmax), use $(x_i-max) - \log \sum e^{x_j-m}$.
        Do not \log e^{x_i-max}, this is numerically unstable when $x_i-max$ is
        small, which will cause an underflow in exp(), then log() might results
        -Inf. Finally, times it with zero results a NaN.

        So in numpy, for a single sample, the following code snippet is stable
        enough:

    ```python
        def softmax_cross_entropy(x, y):
            max_x = np.max(x)
            x_shifted = x - max_x
            log_exp = x_shifted - np.log(np.sum(np.exp(x_shifted)))
            return -np.sum(log_exp * y)
    ```
    """

    ctx = tf.distribute.get_replica_context()
    assert ctx is not None, "forward should be in replica context"

    BC = tf.shape(logits)
    batch_size = BC[0]
    # Assuming logits are equally partitioned, thus this value is the same across all replicas
    num_classes_local = BC[1]

    if labels.shape.rank == 2:
        # NOTE: we should avoid this type of label if we indeed have this large amount of classes.
        tf.assert_equal(BC, tf.shape(labels))
        labels_onehot_local = tf.cast(labels, dtype=tf.float32)
    elif labels.shape.rank == 1:
        tf.assert_equal(tf.shape(labels), batch_size)
        tmp = labels - tf.cast(num_classes_local * ctx.replica_id_in_sync_group, dtype=labels.dtype)
        labels_onehot_local = tf.one_hot(tmp, depth=num_classes_local, dtype=tf.float32)
    else:
        raise RuntimeError("wrong label rank")

    # NOTE: collective communication library's reduce ops are for reducing gradients, The reduction involves large amount of data
    #   exchange. xCCL implement special communication algorithms, to avoid routing thoes data over channels with small bandwidth.
    #   e.g. they first reduce locally and then globally.

    logits_no_grad = tf.stop_gradient(logits)
    max_local = tf.reduce_max(logits_no_grad, axis=1, keepdims=True)
    max_global = ctx.merge_call(lambda _, v: tf.reduce_max(v.values, axis=0), args=(max_local,))
    logits_shifted = logits_no_grad - max_global

    e_to_the_xi = tf.math.exp(logits_shifted)
    sum_local = tf.reduce_sum(e_to_the_xi, axis=1, keepdims=True)
    sum_global = ctx.all_reduce(tf.distribute.ReduceOp.SUM, sum_local)

    # this is log prob, prob in [0, 1], log prob in (-inf, 0]
    pred_local = logits_shifted - tf.math.log(sum_global)
    pred_local = tf.clip_by_value(pred_local, -1e35, 0.0)

    loss_local = tf.reduce_sum(-labels_onehot_local * pred_local, axis=-1)
    loss_global = ctx.all_reduce(tf.distribute.ReduceOp.SUM, loss_local)

    def grad_fn(grad):
        # gradient of softmax cross entropy is particularly simple:
        # -(label - prob)
        return ((e_to_the_xi / sum_global) - labels_onehot_local) * tf.expand_dims(grad, axis=-1), None

    return loss_global, grad_fn
