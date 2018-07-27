import argparse
import datetime

import numpy as np
import tensorflow as tf


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

    return parser


def get_tensor(batch_size):
    return tf.random_normal([batch_size, 500, 120], dtype=tf.float64)


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.iterations % args.batch_size != 0:
        raise RuntimeError('Iterations must be multiple of batch size (i.e. 200 and 2, 200 and 20)')

    x = tf.placeholder(shape=[None, 500, 120], dtype=tf.float64)
    nn = tf.layers.dense(x, 1024, activation=tf.nn.sigmoid)
    nn = tf.layers.dense(nn, 1020, activation=tf.nn.sigmoid)
    nn = tf.layers.dense(nn, 1000, activation=tf.nn.sigmoid)
    nn = tf.layers.dense(nn, 1024, activation=tf.nn.sigmoid)
    nn = tf.layers.dense(nn, 120, activation=tf.nn.sigmoid)

    cost = tf.reduce_mean((nn - x)**2)
    optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)
    init = tf.global_variables_initializer()

    iters = int(args.iterations / args.batch_size)
    summ = 0

    print('Running benchmark...')
    with tf.Session() as sess:
        sess.run(init)
        t = datetime.datetime.now()
        data = get_tensor(args.batch_size)
        tensor = sess.run(data)
        feed_dict = {x: tensor}
        for step in range(iters):
            _, val = sess.run(
                [optimizer, cost],
                feed_dict=feed_dict
            )
            if (step+1) % 5 == 0:
                logging.info("step: {}, value: {}".format(step+1, val))

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


