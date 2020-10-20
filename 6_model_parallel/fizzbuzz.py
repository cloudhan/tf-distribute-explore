import math
import argparse
import numpy as np
import tensorflow as tf

import distribute_extension
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

NUM_HIDDEN = 200


class FizzBuzz:
    def __init__(self):
        self.fbe = FizzBuzzExtended(args.max_number, args.num_words)
        self.strategy = tf.distribute.MirroredStrategy(devices=tf.config.list_logical_devices("GPU"))

        num_replicas = self.strategy.num_replicas_in_sync
        assert (
            self.fbe.num_output_classes % num_replicas
        ) == 0, f"Cannot evenly distribute classes on devices, num classes: {self.fbe.num_output_classes}, num devices: {num_replicas}"

        self.mono_w_o = None
        self.w_o = tf.random.normal((NUM_HIDDEN, self.fbe.num_output_classes), stddev=0.01)
        self.w_o = self.strategy.experimental_distribute_values_from_function(
            lambda ctx: FizzBuzz.init_partitioned_weights(ctx, self.w_o, axis=1)
        )
        for i, v in enumerate(self.w_o.values):
            print("shape of w_o part", i, v.shape)
        print("Number of devices: {}".format(num_replicas))

        with tf.device(self.w_o.values[0].device):
            self.epoch_loss_acc = tf.Variable(0.0)
            self.epoch_step = tf.Variable(0)

        with self.strategy.scope():
            self.w_h = self.init_mirrored_weights([self.fbe.num_input_digits, NUM_HIDDEN])
            lr = 0.05 * np.array([1, 0.33333, 0.1]) * (args.batchsize / 512)
            lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000, 5000], lr.tolist())
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    @staticmethod
    def init_partitioned_weights(ctx, data, axis):
        shape = list(data.shape)
        assert (shape[axis] % ctx.num_replicas_in_sync) == 0
        part_stride = shape[axis] // ctx.num_replicas_in_sync

        begin = [0] * len(shape)
        begin[axis] = ctx.replica_id_in_sync_group * part_stride
        shape[axis] = part_stride

        return tf.Variable(tf.slice(data, begin, shape))

    def init_mirrored_weights(self, shape):
        return tf.Variable(tf.random.normal(shape, stddev=0.01))

    @staticmethod
    def forward(X, w_h, w_o):
        h = tf.nn.relu(tf.matmul(X, w_h))
        output = distribute_extension.matmul_mp_with_merge(h, w_o)
        return output

    @staticmethod
    def loss(Y, logits):
        Y = tf.stop_gradient(Y)
        l = distribute_extension.softmax_cross_entropy_with_logits(logits, Y)
        # batchsize = tf.cast(tf.shape(Y)[0], dtype=tf.float32)
        # return (1.0 / batchsize) * l
        return l

    @staticmethod
    def train_step(optimizer, X, Y, w_h, w_o):
        ctx = tf.distribute.get_replica_context()
        Y = ctx.merge_call(lambda _, v: tf.concat(v.values, axis=0), args=(Y,))

        with tf.GradientTape(persistent=True) as tape:
            o = FizzBuzz.forward(X, w_h, w_o)
            l = FizzBuzz.loss(Y, o)
        grad_w_o = tape.gradient(l, w_o)
        grad_w_h = tape.gradient(l, w_h)

        lr = optimizer._decayed_lr(tf.float32)
        w_o.assign_add(-lr * grad_w_o)

        optimizer.apply_gradients(zip([grad_w_h], [w_h]))

        return l

    @tf.function
    def run_train_step(self, X, Y):
        """Why the fuck do we need this function?

        1. we need tf graph
        2. strategy.run works with tf.function, tf.function works with PerReplica tf.Variable, but
            a) strategy.run works with tf.function with PerReplica tf.Tensor argument
            b) strategy.run DOES NOT work with tf.function with PerReplica tf.Variable argument!
           shit happeneds.
        3. for 2.b, two solutions:
            a) stop using tf.function for function to be run with strategy.run (thus, this function `run_train_step`)
            b) stop passing PerReplica tf.Variable around!
                this is subtle: strategy.run will reap the tf.Variable out of PerReplica and passing them to correct
                computation replica (function/graph). If we capture the PR Variable, we need to do this by ourself.

        Learn more about strategy.run with DistributedValues, see:
        https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/distribute/mirrored_run.py#L163-L164
        https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/distribute/mirrored_function_strategy.py#L128-L133

        To do it manually, see:
        https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/distribute/distribute_utils.py#L126-L139

        stick with 3.a for now!
        """
        l = self.strategy.run(FizzBuzz.train_step, args=(self.optimizer, X, Y, self.w_h, self.w_o))
        return l

    def train_epoch(self, dataset):
        self.epoch_loss_acc.assign(0.0)
        self.epoch_step.assign(0)
        for batch in dataset:
            self.epoch_step.assign_add(1)
            l = self.run_train_step(batch["data"], batch["label"])
            self.epoch_loss_acc.assign_add(tf.reduce_sum(l.values[0]))

    def train(self, dataset, trX, trY):
        for epoch in range(args.num_epoches):
            self.train_epoch(dataset)
            train_loss = self.epoch_loss_acc / tf.cast(self.epoch_step, dtype=tf.float32)
            if (epoch % 20) == 0:
                self.prepare_predict()
                acc = np.mean(trY == fb.predict(trX))
                tf.print("Epoch:", epoch, "train loss:", train_loss, "acc:", acc)
            else:
                tf.print("Epoch:", epoch, "train loss:", train_loss)

    def prepare_predict(self):
        self.mono_w_o = tf.concat(self.w_o.values, axis=1)

    def predict(self, X):
        return tf.argmax(tf.matmul(tf.nn.relu(tf.matmul(X, self.w_h)), self.mono_w_o), axis=1)


fb = FizzBuzz()

# Let's overfitting it!
trX = np.array([fb.fbe.binary_encode(i) for i in range(args.max_number)]).astype(np.float32)
trY = np.array([fb.fbe.sparse_encode(i) for i in range(args.max_number)]).astype(np.int64)


ds = tf.data.Dataset.from_tensor_slices({"data": trX, "label": trY}).cache()
ds = ds.shuffle(buffer_size=args.max_number)
ds = ds.batch(args.batchsize, drop_remainder=True)
dist_ds = fb.strategy.experimental_distribute_dataset(ds)


fb.train(dist_ds, trX, trY)
