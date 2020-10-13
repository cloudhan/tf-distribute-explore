import tensorflow as tf


@tf.function
def my_relu(x):
    @tf.custom_gradient
    def forward(x):
        def grad_fn(grad):
            # return tf.ones_like(x) # <-- uncomment this line to change the grad to ones
            return tf.zeros_like(x)

        return tf.nn.relu(tf.stop_gradient(x)), grad_fn

    return forward(x)


if __name__ == "__main__":
    x = tf.constant([-1.0, 0.0, 1.0, 2.0, 3.0])
    with tf.GradientTape() as tape:
        # NOTE: by default, tape only watch tf.Variable, to compute gradient w.r.t. tensor, watch it.
        tape.watch(x)

        y = my_relu(x)

    print(y)
    print(tape.gradient(y, [x]))
