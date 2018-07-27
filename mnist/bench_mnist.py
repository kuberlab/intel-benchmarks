import argparse
import datetime

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


logging = tf.logging
logging.set_verbosity(logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of benchmark iterations',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size',
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory.',
    )

    return parser


def get_batch(batch_size):
    images = tf.random_normal([batch_size, 784], dtype=tf.float32)
    labels = tf.random_normal([batch_size, 10], dtype=tf.float32)
    return images, labels


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main():
    parser = get_parser()
    args = parser.parse_args()

    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    
    if args.iterations % args.batch_size != 0:
        raise RuntimeError('Iterations must be multiple of batch size (i.e. 200 and 2, 200 and 20)')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    iters = int(args.iterations / args.batch_size)
    summ = 0

    print('Running benchmark...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = datetime.datetime.now()
        for step in range(iters):
            batch = mnist.train.next_batch(args.batch_size)
            tensor = batch[0]
            labels = batch[1]
            #data, labels = get_batch(args.batch_size)
            #tensor = sess.run(data)
            #labels = sess.run(labels)

            train_step.run(feed_dict={x: tensor, y_: labels, keep_prob: 0.5})

            if (step+1) % 100 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: tensor, y_: labels, keep_prob: 1.0}
                )
                logging.info('step %d, training accuracy %g' % (step+1, train_accuracy))

    delta = datetime.datetime.now() - t
    msecs = delta.total_seconds() * 1000
    print('-' * 50)
    print('Benchmark result:')
    print('%s ops, (%.3f op/s)' % (
        args.iterations, args.iterations / delta.total_seconds()
    ))
    print('Total time: %.3fs' % (msecs / 1000.))
    print('-' * 50)


if __name__ == '__main__':
    main()


