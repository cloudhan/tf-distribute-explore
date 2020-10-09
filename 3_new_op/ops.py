import tensorflow as tf


@tf.function
def softmax_cross_entropy_with_logits(logits, labels):
    @tf.custom_gradient
    def forward(logits, labels):
        """reimplement `tf.nn.softmax_cross_entropy_with_logits`

        Args:
            logits: see https://stackoverflow.com/a/52223970/2091555
            labels: index of corresponding label, one-hot label [[1,0,0], [0,1,0], [0,0,1]]
                    or [0, 1, 2], be of shape [batchsize]
        """
        # logits should be a dense matrix of shape [batch_size, num_classes]
        BC = tf.shape(logits)
        batch_size = BC[0]
        num_classes = BC[1]

        if labels.shape.rank == 2:
            tf.assert_equal(tf.rank(labels), 2)
            tf.assert_equal(BC, tf.shape(labels))
            labels_onehot = tf.cast(labels, dtype=tf.float32)
        elif labels.shape.rank  == 1:
            tf.assert_equal(tf.rank(labels), 1)
            tf.assert_equal(tf.shape(labels), batch_size)
            labels_onehot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        else:
            raise RuntimeError("wrong label rank")

        # won't bother to implement a stable softmax here, for now.
        pred = tf.nn.softmax(tf.stop_gradient(logits))
        loss_cross_entropy = tf.reduce_sum(-labels_onehot * tf.math.log(pred), axis=-1)

        def grad_fn(grad):
            # return tf.zeros_like(logits), None # <-- uncomment to see grad_fn is actually used in the computation

            # NOTE: forward compute loss, input grad should be identity! ignore it for now.

            # NOTE: The function we are defining has two inputs, thus, the gradient function
            #  needs to return 2 values, and the second input is label, its gradient is useless
            #  and is generally not required. so we simply returns None for it.
            return (pred - labels_onehot), None

        return loss_cross_entropy, grad_fn

    return forward(logits, labels)


if __name__ == "__main__":
    logits = tf.constant([
        [1.0, 2, 3, 5],
        [7, 11, 13, 17],
        [19, 23, 29, 31],
    ]) # 3 samples, 4 classes

    logits = tf.Variable(logits) # make tf record gradients!
    labels = tf.constant([3,2,1])
    # tf.assert_equal(tf.rank(labels), 2)

    with tf.GradientTape() as tape1:
        ref = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    with tf.GradientTape() as tape2:
        my = softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    ref_grads = tape1.gradient(ref, [logits])
    my_grads = tape2.gradient(my, [logits])

    print("forward:")
    print(ref)
    print(my)

    print("grad:")
    print(ref_grads[0])
    print(my_grads[0])

    print("forward diff:", ref - my)
    print("grad diff:", ref_grads[0] - my_grads[0])

    # so the diff is small enough, we are good for now!
    # the problem remains here is:
    #   How do we
    #   INCORPORATE custom op/function, which have custom gradient (function),
    #   WITH custom distribute strategy (replication of computation).
