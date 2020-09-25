import numpy as np
import tensorflow as tf

from fizzbuzz_utils import FizzBuzzExtended

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

MAX_NUMBER = 2000
NUM_WORDS = 4
fbe = FizzBuzzExtended(MAX_NUMBER, NUM_WORDS)

# Let's overfitting it!
trX = np.array([fbe.binary_encode(i) for i in range(MAX_NUMBER)]).astype(np.float32)
trY = np.array([fbe.sparse_encode(i) for i in range(MAX_NUMBER)]).astype(np.int64)

NUM_HIDDEN = 200
w_h = init_weights([fbe.num_input_digits, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, fbe.num_output_classes])

BATCH_SIZE = 128
NUM_EPOCHES = 10000

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000, 5000], [0.5, 0.1, 0.01])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

for epoch in range(NUM_EPOCHES):
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]

    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        l = train_step(optimizer, trX[start:end], w_h, w_o, trY[start:end])

    print(epoch, np.mean(trY == model_pred(trX, w_h, w_o)))

numbers = np.arange(1, MAX_NUMBER)
test_X = np.transpose(fbe.binary_encode(numbers)).astype(np.float32)
test_Y = model_pred(test_X, w_h, w_o)
output = np.vectorize(fbe.decode)(numbers, test_Y)
print(list(output))
