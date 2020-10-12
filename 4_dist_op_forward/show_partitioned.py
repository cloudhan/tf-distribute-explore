import tensorflow as tf
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy

cc = CollectiveAllReduceStrategy()



shape = (10, 10)

def create_value(ctx):
    return tf.Variable(tf.random.normal(shape=shape))

m = cc.experimental_distribute_values_from_function(create_value)

with cc.scope():
    # NOTE: only variable is controlled by strategy
    v = tf.Variable(tf.random.normal((10, 1)))


@tf.function
def run():
    def pmatrix_mul_mvector(m, v):
        ctx = tf.distribute.get_replica_context()
        assert ctx is not None
        return tf.matmul(m, v)

    return cc.run(pmatrix_mul_mvector, args=(m, v))


dist_result = tf.concat(run().values, axis=0)

mono_m = tf.concat(m.values, axis=0)
mono_v = v.values[0]
mono_result = tf.matmul(mono_m, mono_v)

print(tf.concat((dist_result, mono_result), axis=1))
print(tf.reduce_sum(tf.abs(dist_result - mono_result)))
