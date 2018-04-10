# Architektura sieci:
# 5 warstw
# 3 konwolucyjne
# na wejsciu 28 na 28 na 1 (rozmiar obrazka)
# output 10 (cyfry 0 - 9)
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

if __name__ == "__main__":
    tf.set_random_seed(0)
    # ciekawa funkcja z tensorflow :P
    mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)
    # warstwa wejsciowa
    inputLayer = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # warstwa wyjściowa
    outputLayer = tf.placeholder(tf.float32, [None, 10])
    #dropout = 1 (test), 0.75 (train)
    propabilityKeeper = tf.placeholder(tf.float32)

    # layers sizes
    firstLayer = 4
    secondLayer = 8
    thirdLayer = 16

    fullyConnectedLayer = 256

    # weights - initialized with random values from normal distribution mean=0, stddev=0.1

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, firstLayer], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([firstLayer], stddev=0.1))
    W2 = tf.Variable(tf.truncated_normal([3, 3, firstLayer, secondLayer], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([secondLayer], stddev=0.1))
    W3 = tf.Variable(tf.truncated_normal([3, 3, secondLayer, thirdLayer], stddev=0.1))
    b3 = tf.Variable(tf.truncated_normal([thirdLayer], stddev=0.1))
    W4 = tf.Variable(tf.truncated_normal([7 * 7 * thirdLayer, fullyConnectedLayer], stddev=0.1))
    b4 = tf.Variable(tf.truncated_normal([fullyConnectedLayer], stddev=0.1))
    W5 = tf.Variable(tf.truncated_normal([fullyConnectedLayer, 10], stddev=0.1))
    b5 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

    XX = tf.reshape(inputLayer, [-1, 7 * 7 * thirdLayer])

    # Sam model sieci:
    # max pool - max z kwadratów 2 na 2
    maxPoolFactor = 2
    strideFactor = 1

    Y1 = tf.nn.relu(tf.nn.conv2d(inputLayer, W1, strides=[1, strideFactor, strideFactor, 1], padding='SAME') + b1)
    #  max pool między pierwszą a drugą
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, strideFactor, strideFactor, 1], padding='SAME') + b2)
    Y2 = tf.nn.max_pool(Y2, ksize=[1, maxPoolFactor, maxPoolFactor, 1], strides=[1, maxPoolFactor, maxPoolFactor, 1], padding='SAME')

    # max pool między drugą a trzecią
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, strideFactor, strideFactor, 1], padding='SAME') + b3)
    Y3 = tf.nn.max_pool(Y3, ksize=[1, maxPoolFactor, maxPoolFactor, 1], strides=[1, maxPoolFactor, maxPoolFactor, 1], padding='SAME')

    # z wielu wymiarów ostatniej warstwy konwolucyjnej do jednego
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * thirdLayer])

    # warstwa 4 = YY * W4
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
    # Y4 = tf.nn.dropout(Y4, pkeep)
    Ylogits = tf.matmul(Y4, W5) + b5
    # liczymy prawdopodobieństwa
    Y = tf.nn.softmax(Ylogits)


    # mamy już połączone warstwy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=outputLayer)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(outputLayer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # do magicznego trenowania
    learning_rate = 0.004
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


