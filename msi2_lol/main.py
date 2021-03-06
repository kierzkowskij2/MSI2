# Architektura sieci:
# 5 warstw
# 3 konwolucyjne
# na wejsciu 28 na 28 na 1 (rozmiar obrazka)
# output 10 (cyfry 0 - 9)
import tensorflow as tf
import plotly
from plotly.graph_objs import Scatter, Layout
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

if __name__ == "__main__":


    # augmentacja danych -> przesuniecia, obroty (małe)

    iterations_number = 500
    displays_number = 50
    batch_size = 100

    tf.set_random_seed(0)
    # ciekawa funkcja z tensorflow :P
    mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)
    # warstwa wejsciowa
    inputLayer = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # warstwa wyjściowa
    outputLayer = tf.placeholder(tf.float32, [None, 10])

    # 1.0 w przypadku trenujacym i 0.75 w przypadku testowym
    pkeep = tf.placeholder(tf.float32)

    # layers sizes
    firstLayer = 4
    secondLayer = 8
    thirdLayer = 16

    fullyConnectedLayer = 256

    firstFilterSize = 7
    secondFilterSize = 5
    thirdFilterSize = 3
    fourthFilterSize = 7

    W1 = tf.Variable(tf.truncated_normal([firstFilterSize, firstFilterSize, 1, firstLayer], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([firstLayer], stddev=0.1))
    W2 = tf.Variable(tf.truncated_normal([secondFilterSize, secondFilterSize, firstLayer, secondLayer], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([secondLayer], stddev=0.1))
    W3 = tf.Variable(tf.truncated_normal([thirdFilterSize, thirdFilterSize, secondLayer, thirdLayer], stddev=0.1))
    b3 = tf.Variable(tf.truncated_normal([thirdLayer], stddev=0.1))
    W4 = tf.Variable(tf.truncated_normal([fourthFilterSize * fourthFilterSize * thirdLayer, fullyConnectedLayer], stddev=0.1))
    b4 = tf.Variable(tf.truncated_normal([fullyConnectedLayer], stddev=0.1))
    W5 = tf.Variable(tf.truncated_normal([fullyConnectedLayer, 10], stddev=0.1))
    b5 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

    XX = tf.reshape(inputLayer, [-1, fourthFilterSize * fourthFilterSize * thirdLayer])

    # Sam model sieci:
    # max pool - max z kwadratów 2 na 2 (mozna ustawic inaczej)
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

    # warstwa 4 = YY * W4 zwykle mnozonko macierzy
    # chcemy jeden wymiar
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
    Ylogits = tf.matmul(Y4, W5) + b5
    # liczymy prawdopodobieństwa
    Y = tf.nn.softmax(Ylogits)


    # mamy już połączone warstwy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=outputLayer)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(outputLayer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # do magicznego trenowania
    learning_rate = 0.0035
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # potrzebne zmienne
    init = tf.global_variables_initializer()

    points = list()
    train_losses = list()
    train_accuracies = list()
    test_losses = list()
    test_accuracies = list()

    saver = tf.train.Saver()

    # sumowanie wszystkich wag i biasów

    allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
    allbiases = tf.concat([tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1])], 0)

    # pętla ucząca
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations_number + 1):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)

            if i % (iterations_number/displays_number) == 0:
                train_accuracy, train_loss, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases],
                                                   feed_dict={inputLayer: batch_X, outputLayer: batch_Y, pkeep: 1.0})

                test_accuracy, test_loss = sess.run([accuracy, cross_entropy],
                                             feed_dict={inputLayer: mnist.test.images, outputLayer: mnist.test.labels, pkeep: 1.0})

                print("#{} Train accuracy = {} , Train loss = {} Test accuracy = {} , Test loss={}".format(i, train_accuracy, train_loss, test_accuracy,
                                                                                     test_loss))
                points.append(i)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)

            # wlasciwe trenowanie
            sess.run(train_step, feed_dict={inputLayer: batch_X, outputLayer: batch_Y, pkeep: 0.75})

            # wyswietlenie jak zmienia podczas uczenia (na podstawie list ze zmianami)
            # moga byc dwa wykresiki (dla trenowania i dla testowania)

            if(i >= iterations_number):
                x = points
                y1 = train_losses
                y2 = train_accuracies
                y3 = test_losses
                y4 = test_accuracies


                plotly.offline.plot({
                    "data": [Scatter(x=x, y=y1)],
                    "layout": Layout(title="Train losses")
                }, filename = 'train_losses.html')

                plotly.offline.plot({
                    "data": [Scatter(x=x, y=y2)],
                    "layout": Layout(title="Train accuracies")
                }, filename = 'train_accuracies.html')

                plotly.offline.plot({
                    "data": [Scatter(x=x, y=y3)],
                    "layout": Layout(title="Test losses")
                }, filename = 'test_losses.html')

                plotly.offline.plot({
                    "data": [Scatter(x=x, y=y4)],
                    "layout": Layout(title="Test accuracies")
                }, filename = 'test_accuracies.html')


