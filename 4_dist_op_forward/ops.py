import tensorflow as tf


def softmax_cross_entropy_with_logits(logits, labels):
    """Compute softmax loss partitioned on multiple devices. Assuming they are equally partitioned.

    Args:
        logits: PerReplica of Tensor (a.k.a, partitioned), e.g., If we have a PerReplica Value (with 4 replica) of
            Tensors be of shape 128 * 250_000. Then our batchsize is 128, and we are doing 1 million classes
            classification (250000 * 4 replicas), with each replica handles only 250000 classes.
            This is called model parallelism.
        labels: MirroredVariable of Tensor
    """

    @tf.custom_gradient
    def forward(logits, labels):
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
            labels_onehot_local = tf.one_hot(labels - num_classes_local * ctx.replica_id_in_sync_group, depth=num_classes_local, dtype=tf.float32)
            tf.print(labels - num_classes_local * ctx.replica_id_in_sync_group)
        else:
            raise RuntimeError("wrong label rank")

        # NOTE: collective communication library's reduce ops are for reducing gradients, they cannot reduce locally
        #   and then globally. Thus, the reduction involves large amount of data exchange. xCCL implement special
        #   algorithms, to avoid routing thoes data over channels with small bandwidth.
        # So we won't use it.

        logits_no_grad = tf.stop_gradient(logits)
        max_local = tf.reduce_max(logits_no_grad, axis=1, keepdims=True)
        max_global = ctx.merge_call(lambda _, v: tf.reduce_max(v.values, axis=0), args=(max_local,))
        e_to_the_xi = tf.math.exp(logits_no_grad - max_global)
        sum_local = tf.reduce_sum(e_to_the_xi, axis=1, keepdims=True)
        sum_global = ctx.merge_call(lambda _, v: tf.reduce_sum(v.values, axis=0), args=(sum_local,))

        pred_local = tf.math.log(e_to_the_xi / sum_global) # this is log prob, prob in [0, 1], log_prob in (-inf, 0]
        pred_local = tf.clip_by_value(pred_local, -1e35, 1e-35)
        tf.print(labels_onehot_local)
        loss_local = tf.reduce_sum(-labels_onehot_local * pred_local, axis=-1)
        loss_global = ctx.merge_call(lambda _, v: tf.reduce_sum(v.values, axis=0), args=(loss_local,))

        def grad_fn(grad):
            # NOTE: NotImplemented, for now!
            return None, None

        return loss_global, grad_fn

    return forward(logits, labels)


if __name__ == "__main__":
    dp = tf.distribute.MirroredStrategy()
    num_replica = dp.num_replicas_in_sync

    batchsize = 16
    num_class_pr = 4 # number of classes per replica
    logits = tf.range(batchsize * num_class_pr * num_replica, dtype=tf.float32)
    logits_partitioned = tf.reshape(logits, shape=(batchsize, num_replica, num_class_pr))

    def partition_logits(ctx):
        return logits_partitioned[:, ctx.replica_id_in_sync_group, :]

    logits_partitioned = dp.experimental_distribute_values_from_function(partition_logits)
    logits = tf.reshape(logits, shape=(batchsize, -1))

    labels = tf.math.mod(tf.range(batchsize), num_class_pr * num_replica)

    with dp.scope():
        labels_mirrored = tf.Variable(labels)

    ref = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    my = dp.run(softmax_cross_entropy_with_logits, args=(logits_partitioned, labels_mirrored))
    # my = tf.concat(my.values, axis=1)

    print("forward:")
    print(ref)
    print(my)

    # print("forward diff:", ref - my)
