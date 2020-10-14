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
        labels_onehot_local = tf.one_hot(
            labels - num_classes_local * ctx.replica_id_in_sync_group, depth=num_classes_local, dtype=tf.float32
        )
    else:
        raise RuntimeError("wrong label rank")

    # NOTE: collective communication library's reduce ops are for reducing gradients, The reduction involves large amount of data
    #   exchange. xCCL implement special communication algorithms, to avoid routing thoes data over channels with small bandwidth.
    #   e.g. they first reduce locally and then globally.
    # Our data exchange is small, so not using it makes no problem.

    logits_no_grad = tf.stop_gradient(logits)
    max_local = tf.reduce_max(logits_no_grad, axis=1, keepdims=True)
    max_global = ctx.merge_call(lambda _, v: tf.reduce_max(v.values, axis=0), args=(max_local,))
    logits_shifted = logits_no_grad - max_global

    e_to_the_xi = tf.math.exp(logits_shifted)
    sum_local = tf.reduce_sum(e_to_the_xi, axis=1, keepdims=True)
    sum_global = ctx.merge_call(lambda _, v: tf.reduce_sum(v.values, axis=0), args=(sum_local,))

    # this is log prob, prob in [0, 1], log prob in (-inf, 0]
    pred_local = logits_shifted - tf.math.log(sum_global)
    pred_local = tf.clip_by_value(pred_local, -1e35, 0.0)

    loss_local = tf.reduce_sum(-labels_onehot_local * pred_local, axis=-1)
    loss_global = ctx.merge_call(lambda _, v: tf.reduce_sum(v.values, axis=0), args=(loss_local,))

    def grad_fn(grad):
        # gradient of softmax cross entropy is particularly simple:
        # -(label - prob)
        return (e_to_the_xi / sum_global) - labels_onehot_local, None

    return loss_global, grad_fn


if __name__ == "__main__":
    dp = tf.distribute.MirroredStrategy()
    num_replica = dp.num_replicas_in_sync

    batchsize = 64
    num_class_pr = 25  # number of classes per replica
    logits = tf.range(batchsize * num_class_pr * num_replica, dtype=tf.float32)
    logits_partitioned = tf.reshape(logits, shape=(batchsize, num_replica, num_class_pr))

    def partition_logits(ctx):
        return logits_partitioned[:, ctx.replica_id_in_sync_group, :]

    logits_partitioned = dp.experimental_distribute_values_from_function(partition_logits)
    logits = tf.reshape(logits, shape=(batchsize, -1))

    labels = tf.math.mod(tf.range(batchsize), num_class_pr * num_replica)

    with dp.scope():
        labels_mirrored = tf.Variable(labels)

    with tf.GradientTape() as tape:
        tape.watch(logits)
        ref = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    ref_grad, = tape.gradient(ref, [logits])

    # NOTE: softmax_cross_entropy_with_logits is not a tf.function, the bug will
    # not be triggered. for simplicity here, we will not wrap the whole
    # computation in a tf.function.
    def forward_backward(logits, labels):  # def step(...):
        with tf.GradientTape() as tape:
            tape.watch(logits)
            ret = softmax_cross_entropy_with_logits(logits, labels)

        assert tf.distribute.get_replica_context() is not None
        grad, = tape.gradient(ret, [logits])
        return ret, grad

    my, my_grad = dp.run(forward_backward, args=(logits_partitioned, labels_mirrored))

    print("forward:")
    print(ref)
    print(my)

    forward_diff = ref - my.values[0]
    print("forward diff:", forward_diff)

    my_grad = tf.concat(my_grad.values, axis=1)

    print("backward:")
    print(ref_grad)
    print(my_grad)

    backward_diff = ref_grad - my_grad
    print("backward diff:", backward_diff)

    print("forward abs error:", tf.reduce_sum(tf.abs(forward_diff)))
    print("backward abs error:", tf.reduce_sum(tf.abs(backward_diff)))

    # So the difference is small!
