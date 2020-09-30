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
    ctx = tf.distribute.get_replica_context()
    with tf.GradientTape() as tape:
        o = model(X, w_h, w_o)
        l = loss(Y, o)
    grads = tape.gradient(l, [w_h, w_o])
    optimizer.apply_gradients(zip(grads, [w_h, w_o]))
    return l


@tf.function
def train_loop(dataset, optimizer, w_h, w_o, epoch_loss):
    strategy = tf.distribute.get_strategy()
    strategy.run(lambda: epoch_loss.assign(0.0))
    num_steps = 0.0
    for batch in dataset:
        num_steps += 1.0
        l = strategy.run(train_step, args=(optimizer, batch["data"], w_h, w_o, batch["label"]))
        # strategy.run(lambda: epoch_loss.assign_add(l))
        strategy.run(lambda acc, v: acc.assign_add(v), args=(epoch_loss, l))

    strategy.run(lambda: epoch_loss.assign(epoch_loss / num_steps))

fbe = FizzBuzzExtended(args.max_number, args.num_words)

# Let's overfitting it!
trX = np.array([fbe.binary_encode(i) for i in range(args.max_number)]).astype(np.float32)
trY = np.array([fbe.sparse_encode(i) for i in range(args.max_number)]).astype(np.int64)

dp = tf.distribute.MirroredStrategy(devices=tf.config.list_logical_devices("GPU"))
print ('Number of devices: {}'.format(dp.num_replicas_in_sync))

ds = tf.data.Dataset.from_tensor_slices({"data": trX, "label": trY}).cache()
ds = ds.shuffle(buffer_size=args.max_number)
ds = ds.batch(args.batchsize, drop_remainder=True)
dist_ds = dp.experimental_distribute_dataset(ds)

# CHECK dataset, this shows the DistributedValues' structure
# with dp.scope():
#     for epoch in range(args.num_epoches):
#         for batches in dist_ds:
#             for batch in zip(batches["data"]._values, batches["label"]._values):
#                 data, label = batch
#                 data = data.numpy()
#                 label = label.numpy()
#                 for i in range(len(label)):
#                     number = fbe.binary_decode(data[i].tolist())
#                     label1 = label[i]
#                     label2 = fbe.sparse_encode(number)
#                     assert label1 == label2
#     exit()

with dp.scope():
    # record avg loss in a epoch
    epoch_loss = tf.Variable(0.0, tf.float32)

    NUM_HIDDEN = 200
    w_h = init_weights([fbe.num_input_digits, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, fbe.num_output_classes])

    # learning rate is critical!
    lr = 0.05 * np.array([1, 0.33333, 0.1]) * (args.batchsize / 512)
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000, 5000], lr.tolist())
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    for epoch in range(args.num_epoches):
        train_loop(dist_ds, optimizer, w_h, w_o, epoch_loss)
        print(f"Epoch: {epoch}: acc: {np.mean(trY == model_pred(trX, w_h, w_o))}, pre-replica training loss: {list(map(lambda x: x.numpy(), epoch_loss._values))}")

numbers = np.arange(1, args.max_number)
test_X = np.transpose(fbe.binary_encode(numbers)).astype(np.float32)
test_Y = model_pred(test_X, w_h, w_o)
output = np.vectorize(fbe.decode)(numbers, test_Y)
print(list(output))
