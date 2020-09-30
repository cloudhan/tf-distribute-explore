import argparse
import numpy as np
import tensorflow as tf

from fizzbuzz_utils import FizzBuzzExtended


parser = argparse.ArgumentParser()
parser.add_argument("--max-number", "-n", type=int, default=2000)
parser.add_argument("--num-words", "-w", type=int, default=4)
parser.add_argument("--batchsize", "-b", type=int, default=128)
parser.add_argument("--num-epoches", "-e", type=int, default=10000)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

if args.debug:
    tf.debugging.set_log_device_placement(True)


def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))


@tf.function
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    output = tf.matmul(h, w_o)
    return output


@tf.function
def model_pred(X, w_h, w_o):
    return tf.argmax(input=model(X, w_h, w_o), axis=1)


@tf.function
def loss(Y, pred):
    Y = tf.stop_gradient(Y)
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=Y)
    return tf.reduce_mean(l)


@tf.function
def train_step(optimizer, X, w_h, w_o, Y):
    # NOTE: in mirrored strategy scope, with replica ctx
    assert not tf.distribute.in_cross_replica_context(), "We should be in replica context"

    with tf.GradientTape() as tape:
        o = model(X, w_h, w_o)
        l = loss(Y, o)
    grads = tape.gradient(l, [w_h, w_o])

    ctx = tf.distribute.get_replica_context()
    # NOTE: these grads are PerReplica values
    # tf.print("Replica id:", ctx.replica_id_in_sync_group, "of", ctx.num_replicas_in_sync)
    # tf.print(grads)

    @tf.function
    def merge_fn(strategy, dist_v):
        # NOTE: with merge_call, we enter the cross replica ctx again
        assert tf.distribute.in_cross_replica_context()
        # tf.print(strategy) # NOTE: print shows we are in the MirroredStrategy
        v = strategy.reduce(tf.distribute.ReduceOp.MEAN, dist_v, axis=None)
        return v

    # NOTE: this is unnecessarily complicated way to do all reduction, just for demonstration purpose.
    # NOTE: tf.keras.optimizers.Optimizer.apply_gradients will automatic do this for us (again),
    #   but mean all_reduce mean grads are still original mean grads anyway ;)
    mean_grads = ctx.merge_call(merge_fn, args=(grads,))
    optimizer.apply_gradients(zip(mean_grads, [w_h, w_o]))
    return l


fbe = FizzBuzzExtended(args.max_number, args.num_words)

# Let's overfitting it!
trX = np.array([fbe.binary_encode(i) for i in range(args.max_number)]).astype(np.float32)
trY = np.array([fbe.sparse_encode(i) for i in range(args.max_number)]).astype(np.int64)

ds = tf.data.Dataset.from_tensor_slices({"data": trX, "label": trY}).cache()
ds = ds.batch(args.batchsize, drop_remainder=True)

# dp means data parallel
dp = tf.distribute.MirroredStrategy(devices=tf.config.list_logical_devices("GPU"))
print ('Number of devices: {}'.format(dp.num_replicas_in_sync))


# NOTE: no strategy scope is entered, we are in the default single replica ctx
with dp.scope():
    # NOTE: in mirrored strategy scope, with cross replica ctx
    assert tf.distribute.in_cross_replica_context(), "We should be in cross replica context"

    NUM_HIDDEN = 200
    # NOTE: we create tf.Variable, thus, we are in scope
    w_h = init_weights([fbe.num_input_digits, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, fbe.num_output_classes])

    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000, 5000], [0.5, 0.1, 0.01])
    # NOTE: models, optimizers, metrics might create tf.Variable, so they should be in scope
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    dist_ds = dp.experimental_distribute_dataset(ds)


for epoch in range(args.num_epoches):
    # NOTE: the distributed dataset iterator returns PerReplica DistributedValues
    #   thus, it can be used as input of strategy.run
    # for batch in dist_ds:
    #     print(batch)
    #     exit()

    for batch in dist_ds:
        dist_loss = dp.run(train_step, args=(optimizer, batch["data"], w_h, w_o, batch["label"]))

    print(epoch, np.mean(trY == model_pred(trX, w_h, w_o)))
    # exit()


numbers = np.arange(1, args.max_number)
test_X = np.transpose(fbe.binary_encode(numbers)).astype(np.float32)
test_Y = model_pred(test_X, w_h, w_o)
output = np.vectorize(fbe.decode)(numbers, test_Y)
print(list(output))
