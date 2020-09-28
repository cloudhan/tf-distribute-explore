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
    prob = tf.matmul(h, w_o)
    return prob


@tf.function
def model_pred(X, w_h, w_o):
    prob = model(X, w_h, w_o)
    return tf.argmax(input=prob, axis=1)


@tf.function
def loss(Y, pred):
    Y = tf.stop_gradient(Y)
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=Y)
    return tf.reduce_mean(l)


@tf.function
def train_step(optimizer, X, w_h, w_o, Y):
    with tf.GradientTape() as tape:
        prob = model(X, w_h, w_o)
        l = loss(Y, prob)
    grads = tape.gradient(l, [w_h, w_o])
    optimizer.apply_gradients(zip(grads, [w_h, w_o]))
    return l


fbe = FizzBuzzExtended(args.max_number, args.num_words)

# Let's overfitting it!
trX = np.array([fbe.binary_encode(i) for i in range(args.max_number)]).astype(np.float32)
trY = np.array([fbe.sparse_encode(i) for i in range(args.max_number)]).astype(np.int64)

ds = tf.data.Dataset.from_tensor_slices({"data": trX, "label": trY})
ds = ds.batch(args.batchsize, drop_remainder=True)

# dp means data parallel
dp = tf.distribute.MirroredStrategy(devices=tf.config.list_logical_devices("GPU"))
print ('Number of devices: {}'.format(dp.num_replicas_in_sync))

@tf.function
def print_in_ctx(v):
    ctx = tf.distribute.get_replica_context()
    print(ctx)
    if ctx:
        print(dir(ctx))
        print("Value:", ctx.replica_id_in_sync_group, v)
        with ctx:
            print(v)
    else:
        print("ctx is None!")

with dp.scope():
    NUM_HIDDEN = 200
    w_h = init_weights([fbe.num_input_digits, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, fbe.num_output_classes])

    dp.experimental_split_to_logical_devices
    # NOTE: tf.Variable is distribute strategy aware

    print_in_ctx(w_h)
    dp.run(print_in_ctx, args=(w_h,))


exit()

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000, 5000], [0.5, 0.1, 0.01])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)


for epoch in range(args.num_epoches):
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]

    for batch in ds:
        l = train_step(optimizer, batch["data"], w_h, w_o, batch["label"])

    print(epoch, np.mean(trY == model_pred(trX, w_h, w_o)))
    exit()

numbers = np.arange(1, args.max_number)
test_X = np.transpose(fbe.binary_encode(numbers)).astype(np.float32)
test_Y = model_pred(test_X, w_h, w_o)
output = np.vectorize(fbe.decode)(numbers, test_Y)
print(list(output))
