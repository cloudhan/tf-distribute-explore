import numpy as np
import tensorflow as tf

NUM_DIGITS = 10

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)], dtype=np.float32)

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])

# Whatsoever, let's cheat with dataset. My goal is to explore tf.distribute capability.
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i) for i in range(2 ** NUM_DIGITS)])


# @tf.function
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
    l = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y)
    return tf.reduce_mean(l)


@tf.function
def train_step(optimizer, X, w_h, w_o, Y):
    with tf.GradientTape() as tape:
        prob = model(X, w_h, w_o)
        l = loss(Y, prob)
    grads = tape.gradient(l, [w_h, w_o])
    optimizer.apply_gradients(zip(grads, [w_h, w_o]))
    return l


NUM_HIDDEN = 100

# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

BATCH_SIZE = 128
NUM_EPOCHES = 10000

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([8000], [0.05, 0.025])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

for epoch in range(NUM_EPOCHES):
    # Shuffle the data before each training iteration.
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]

    # Train with batches.
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        l = train_step(optimizer, trX[start:end], w_h, w_o, trY[start:end])

    # And print the current accuracy on the training data.
    print(epoch, np.mean(np.argmax(trY, axis=1) == model_pred(trX, w_h, w_o)))

# turn a prediction into a fizz buzz output
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

numbers = np.arange(1, 101)
test_X = np.transpose(binary_encode(numbers, NUM_DIGITS)).astype(np.float32)
test_Y = model_pred(test_X, w_h, w_o)
output = np.vectorize(fizz_buzz)(numbers, test_Y)
print(output)
