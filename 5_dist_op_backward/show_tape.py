import tensorflow as tf
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy

cc = CollectiveAllReduceStrategy()
shape = (10, 10)

# define the op
@tf.custom_gradient
def pmatrix_mul_mvector(m, v):
    """partitioned matrix multiply mirrored vector"""

    def forward(m, v):
        tf.stop_gradient(m)
        tf.stop_gradient(v)
        return tf.matmul(m, v)

    def grad_fn(input_grad):
        grad_m = tf.repeat(tf.transpose(v), axis=0, repeats=m.shape[0])
        grad_v = tf.matmul(tf.transpose(m), input_grad)
        ctx = tf.distribute.get_replica_context()
        grad_v = ctx.all_reduce(tf.distribute.ReduceOp.SUM, grad_v)
        return grad_m, grad_v

    return forward(m, v), grad_fn


def create_value(ctx):
    return tf.Variable(tf.random.normal(shape=shape))


m = cc.experimental_distribute_values_from_function(create_value)
with cc.scope():
    # NOTE: only variable is controlled by strategy
    v = tf.Variable(tf.random.normal((10, 1)))

# build graph
@tf.function
def run():
    # the actual computation to be replicated, we could have put tf.function
    # decorator here, but there is a bug for tf.function to take as input a
    # PerReplica tf.Variable.
    def step(m, v):
        ctx = tf.distribute.get_replica_context()
        assert ctx is not None
        with tf.GradientTape() as tape:
            tape.watch(m)
            tape.watch(v)
            ret = pmatrix_mul_mvector(m, v)

        grads = tape.gradient(ret, [m, v])

        return ret, grads

    return cc.run(step, args=(m, v))


result, (grad_m, grad_v) = run()

dist_result = tf.concat(result.values, axis=0)
dist_grad_m = tf.concat(grad_m.values, axis=0)
dist_grad_v = grad_v.values[0]

mono_m = tf.concat(m.values, axis=0)
mono_v = v.values[0]
with tf.GradientTape() as tape:
    tape.watch(mono_m)
    tape.watch(mono_v)
    mono_result = tf.matmul(mono_m, mono_v)

mono_grad_m, mono_grad_v = tape.gradient(mono_result, [mono_m, mono_v])

print(tf.reduce_sum(tf.abs(dist_result - mono_result)))
print(tf.reduce_sum(tf.abs(dist_grad_m - mono_grad_m)))
print(tf.reduce_sum(tf.abs(dist_grad_v - mono_grad_v)))

# So the diff is small enough!
# And I have all the ingredients to build TF2 native large scale model parallel distributed training task.
