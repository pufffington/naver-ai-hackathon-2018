# -*- coding: utf-8 -*-
"""
Copyright 2018 NAVER Corp.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
import nsml

from random import random

from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess2

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        data1, data2 = preprocess2(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(sigmoid_scores, feed_dict={x1: data1, x2: data2})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다
    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        W = tf.get_variable("W", [output_size, input_size], dtype=input_.dtype)
        b = tf.get_variable("b", [output_size], dtype=input_.dtype)

    return tf.nn.xw_plus_b(input_, tf.transpose(W), b)


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway', name="Highway"):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope=('highway_lin_{0}'.format(idx))))
            t = tf.sigmoid(linear(input_, size, scope=('highway_gate_{0}'.format(idx))) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output

def get_accuracy(data1, data2):
    count = 0
    for i in range(len(data1)):
        if data1[i] == data2[i]:
            count += 1
    return count / len(data1)

def contrastive_loss(y, d, batch_size):
    tmp = y * tf.square(d)
    # tmp= tf.mul(y,tf.square(d))
    tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
    return tf.reduce_sum(tmp + tmp2) / batch_size / 2

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--name', type=str, default='')
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')
    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=15)
    args.add_argument('--batch', type=int, default=256)
    args.add_argument('--strmaxlen', type=int, default=90)
    args.add_argument('--embedding', type=int, default=32)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--shuffle', type=str, default='yes')
    args.add_argument('--reverse_train', type=str, default='yes')

    config = args.parse_args()
    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # tf.reset_default_graph() ##
    # 모델의 specification
    input_size = config.embedding * config.strmaxlen
    output_size = 1
    learning_rate = 0.001  # 0.001
    character_size = 251

    x1 = tf.placeholder(tf.int32, [None, config.strmaxlen])
    x2 = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])

    global_step = tf.Variable(0, trainable=False, name="Global_Step")
    l2_loss = tf.constant(0.0)

    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded_x1 = tf.nn.embedding_lookup(char_embedding, x1)
    embedded_x2 = tf.nn.embedding_lookup(char_embedding, x2)
    embedded_expanded_x1 = tf.expand_dims(embedded_x1, -1)
    embedded_expanded_x2 = tf.expand_dims(embedded_x2, -1)

    pooled_outputs_front = []
    pooled_outputs_behind = []

    for i, filter_size in enumerate([3, 4, 5]):
        with tf.name_scope("conv-filter{0}".format(filter_size)):
            # Convolution Layers
            filter_shape = [filter_size, 32, 1, 256]
            W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[256]), dtype=tf.float32, name="b")
            conv_front = tf.nn.conv2d(
                embedded_expanded_x1,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_front")

            conv_behind = tf.nn.conv2d(
                embedded_expanded_x2,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_behind")

            # Batch Normalization Layer
            conv_bn_front = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv_front, b), is_training=True)
            conv_bn_behind = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv_behind, b), is_training=True)

            # Apply nonlinearity
            conv_out_front = tf.nn.relu(conv_bn_front, name="relu_front")
            conv_out_behind = tf.nn.relu(conv_bn_behind, name="relu_behind")

        with tf.name_scope("pool-filter{0}".format(filter_size)):
            # Maxpooling over the outputs
            pooled_front = tf.nn.max_pool(
                conv_out_front,
                ksize=[1, 90 - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool_front")

            pooled_behind = tf.nn.max_pool(
                conv_out_behind,
                ksize=[1, 90 - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool_behind")

        pooled_outputs_front.append(pooled_front)
        pooled_outputs_behind.append(pooled_behind)

    num_filters_total = 256 * 3
    pool_front = tf.concat(pooled_outputs_front, 3)
    pool_behind = tf.concat(pooled_outputs_behind, 3)
    pool_flat_front = tf.reshape(pool_front, [-1, num_filters_total])
    pool_flat_behind = tf.reshape(pool_behind, [-1, num_filters_total])

    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pool_flat_front, pool_flat_behind)), 1, keep_dims=True))
    distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(pool_flat_front), 1, keep_dims=True)), tf.sqrt(tf.reduce_sum(tf.square(pool_flat_behind), 1, keep_dims=True))))
    distance = tf.reshape(distance, [-1, 1], name="distance")

    pool_flat_combine = tf.concat([pool_flat_front, pool_flat_behind], 1)

    # Fully Connected Layer
    with tf.name_scope("fc"):
        W = tf.Variable(tf.truncated_normal(shape=[num_filters_total * 2, 256], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[256]), dtype=tf.float32, name="b")
        fc = tf.nn.xw_plus_b(pool_flat_combine, W, b)

        fc_bn = tf.contrib.layers.batch_norm(fc, is_training=True)
        fc_out = tf.nn.relu(fc_bn, name="relu")
        fc_drop = tf.nn.dropout(fc_out, 0.5)

    # Fully Connected Layer
    with tf.name_scope("fc1"):
        W1 = tf.Variable(tf.truncated_normal(shape=[256, 16], stddev=0.1), name="W")
        b1 = tf.Variable(tf.constant(0.1, shape=[16]), dtype=tf.float32, name="b")
        fc1 = tf.nn.xw_plus_b(fc_drop, W1, b1)

        fc_bn1 = tf.contrib.layers.batch_norm(fc1, is_training=True)
        fc_out1 = tf.nn.relu(fc_bn1, name="relu")
        fc_drop1 = tf.nn.dropout(fc_out1, 0.5)

    highway = highway(fc_drop1, fc_drop1.get_shape()[1], num_layers=1, bias=0, name="Highway")

    with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal(shape=[16, 1], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[1]), dtype=tf.float32, name="b")
        scores = tf.nn.xw_plus_b(highway, W, b, name="scores")
        scores = tf.multiply(scores, distance)
        bn_scores = tf.contrib.layers.batch_norm(scores)
        sigmoid_scores = tf.sigmoid(bn_scores)
        predictions = sigmoid_scores

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(-(y_ * tf.log(sigmoid_scores)) - (1 - y_) * tf.log(1 - sigmoid_scores))

    # Accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(predictions, tf.cast(y_, dtype="float"))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    # Number of correct predictions
    with tf.name_scope("num_correct"):
        correct = tf.equal(predictions, tf.cast(y_, dtype="float"))
        num_correct = tf.reduce_sum(tf.cast(correct, "float"), name="num_correct")


    optimizer = tf.train.AdamOptimizer(0.001)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)
    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        print(dataset_len)
        one_batch_size = dataset_len // config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data1, data2, labels) in enumerate(_batch_loader(dataset, config.batch)):
                if random() > 0.5:
                    feed_dict = {
                        x1: data1,
                        x2: data2,
                        y_: labels
                    }
                else:
                    feed_dict = {
                        x1: data2,
                        x2: data1,
                        y_: labels
                    }
                _, step, _loss, _sigmoid_scores, _scores, _bn_scores = sess.run([train_op, global_step, loss, sigmoid_scores, scores, bn_scores], feed_dict)
                current_step = tf.train.global_step(sess, global_step)
                clipped = np.array(_sigmoid_scores > config.threshold, dtype=np.int)
                _acc = get_accuracy(labels, clipped)
                print('Batch : ', i + 1, '/', one_batch_size, ', loss: ', float(_loss), ', acc: ', _acc)

                avg_loss += float(_loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss / one_batch_size))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss / one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)