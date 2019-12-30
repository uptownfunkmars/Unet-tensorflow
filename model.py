import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


class Unet(object):

    def __init__(self, batch_norm_decay=0.99, batch_norm_epsilon=1e-3):
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.num_classes = 5

    def model(self, input):
        with tf.variable_scope("encoder"):
            with tf.variable_scope("layer1"):
                w1 = tf.get_variable("weight1", shape=[3, 3, 3, 64])
                w2 = tf.get_variable("weight2", shape=[3, 3, 64, 64])
                conv = self._conv(input, w1, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w2, 1)
                conv = self._relu(conv)
                feature1 = conv
                conv = self._downSample(conv)
                print("feature1 shape : ", feature1.get_shape().as_list())
                # feature1 = conv

            with tf.variable_scope("layer2"):
                w3 = tf.get_variable("weight3", shape=[3, 3, 64, 128])
                w4 = tf.get_variable("weight4", shape=[3, 3, 128, 128])
                conv = self._conv(conv, w3, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w4, 1)
                conv = self._relu(conv)
                feature2 = conv
                conv = self._downSample(conv)
                print("feature2 shape : ", feature2.get_shape().as_list())
                # feature2 = conv

            with tf.variable_scope("layer3"):
                w5 = tf.get_variable("weight5", shape=[3, 3, 128, 256])
                w6 = tf.get_variable("weight6", shape=[3, 3, 256, 256])
                conv = self._conv(conv, w5, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w6, 1)
                conv = self._relu(conv)
                feature3 = conv
                conv = self._downSample(conv)
                print("feature 3 shape : ", feature3.get_shape().as_list())
                # feature3 = conv

            with tf.variable_scope("layer4"):
                w7 = tf.get_variable("weight7", shape=[3, 3, 256, 512])
                w8 = tf.get_variable("weight8", shape=[3, 3, 512, 512])

                conv = self._conv(conv, w7, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w8, 1)
                conv = self._relu(conv)
                feature4 = conv
                conv = self._downSample(conv)
                print("feature4 shape : ", feature4.get_shape().as_list())
                # feature4 = conv

            with tf.variable_scope("layer5"):
                w9 = tf.get_variable("weight9", shape=[3, 3, 512, 1024])
                w10 = tf.get_variable("weight10", shape=[3, 3, 1024, 1024])

                conv = self._conv(conv, w9, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w10, 1)
                conv = self._relu(conv)

        with tf.variable_scope("Decoder"):
            with tf.variable_scope("layer6"):
                w12 = tf.get_variable("weight12", shape=[3, 3, 1024, 512])
                w13 = tf.get_variable("weight13", shape=[3, 3, 1024, 512])

                w14 = tf.get_variable("weight14", shape=[3, 3, 512, 512])

                size = feature4.get_shape().as_list()

                conv = self._upSample(conv, [size[1], size[2]])

                print("L6 conv shape : ", conv.get_shape().as_list())

                conv = self._conv(conv, w12, 1)
                conv = self._relu(conv)
                # print("conv shape : ", conv.get_shape().as_list())
                merge = tf.concat([conv, feature4], axis=3)
                print("merge shape is : ", merge.get_shape().as_list())
                conv = self._conv(merge, w13, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w14, 1)
                conv = self._relu(conv)

            with tf.variable_scope("layer7"):
                w15 = tf.get_variable("weight15", shape=[3, 3, 512, 256])
                w16 = tf.get_variable("weight16", shape=[3, 3, 512, 256])
                w17 = tf.get_variable("weight17", shape=[3, 3, 256, 256])

                size = feature3.get_shape().as_list()

                conv = self._upSample(conv, [size[1], size[2]])

                print("L7 conv shape : ", conv.get_shape().as_list())

                conv = self._conv(conv, w15, 1)
                conv = self._relu(conv)
                merge = tf.concat([conv, feature3], axis=3)

                print("merger shape is : ", merge.get_shape().as_list())

                conv = self._conv(merge, w16, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w17, 1)
                conv = self._relu(conv)

            with tf.variable_scope("layer8"):
                w18 = tf.get_variable("weight18", shape=[3, 3, 256, 128])
                w19 = tf.get_variable("weight19", shape=[3, 3, 256, 128])
                w20 = tf.get_variable("weight20", shape=[3, 3, 128, 128])

                size = feature2.get_shape().as_list()

                conv = self._upSample(conv, [size[1], size[2]])

                print("L8 conv shape : ", conv.get_shape().as_list())

                conv = self._conv(conv, w18, 1)
                conv = self._relu(conv)
                merge = tf.concat([conv, feature2], axis=3)

                print("merge shape is : ", merge.get_shape().as_list())

                conv = self._conv(merge, w19, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w20, 1)
                conv = self._relu(conv)

            with tf.variable_scope("layer9"):
                w21 = tf.get_variable("weight21", shape=[3, 3, 128, 64])
                w22 = tf.get_variable("weight22", shape=[3, 3, 128, 64])
                w23 = tf.get_variable("weight23", shape=[3, 3, 64, 64])
                w24 = tf.get_variable("weight24", shape=[3, 3, 64, 5])
                # w25 = tf.get_variable("weight25", shape=[3, 3, 2, 1],
                #                      initializer=tf.random_normal_initializer())

                size = feature1.get_shape().as_list()

                conv = self._upSample(conv, [size[1], size[2]])

                print("L9 conv shape : ", conv.get_shape().as_list())

                conv = self._conv(conv, w21, 1)
                conv = self._relu(conv)
                merge = tf.concat([conv, feature1], axis=3)
                print("merge shape is : ", merge.get_shape().as_list())
                conv = self._conv(merge, w22, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w23, 1)
                conv = self._relu(conv)
                conv = self._conv(conv, w24, 1)
                # conv = self._relu(conv)
                # conv = self._conv(conv, w25, 1)

        return conv

    def _batchNorm(self, x):
        x_shape = x.get_shape().as_list()
        # channels number
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))
        print("axis is : ", axis)

        beta = tf.get_variable(name='beta', shape=params_shape, initializer=tf.ones_initializer)
        gamma = tf.get_variable(name='gamma', shape=params_shape, initializer=tf.ones_initializer)

        # 为华东更新的变量赋初值 mean=0， variance=1
        moving_mean = tf.get_variable(name='moving_mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable(name='moving_variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)

        tf.add_to_collection('BN_MEAN_VARIANCE', moving_mean)
        tf.add_to_collection('BN_MEAN_VARIANCE', moving_variance)

        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, self._batch_norm_decay, name='MovingAvgMean')
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, self._batch_norm_decay, name='MovingAvgVariance')

        # 滑动指数平均batch均值和方差的操作整合到tensorflow维护的update op集合中去，使每次更新变为自动
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        # tf.cond为tensorflow自带的条件控制语句，若为True执行true_fn，若为False执行False_fn.
        mean, variance = tf.cond(pred=self._is_training,
                                 true_fn=lambda :(mean, variance),
                                 false_fn=lambda : (moving_mean, moving_averages))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, self._batch_norm_epsilon)
        return x

    def _downSample(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def _upSample(self, x, size, align_corners=False, name=None):
        return tf.image.resize_bilinear(x, size, align_corners=align_corners, name=name)

    def _conv(self, x, W, stride, padding="SAME"):
        x = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding=padding)
        return x

    def _relu(self, x):
        return tf.nn.relu(x)

    def _softmax(self, x):
        return tf.nn.softmax(x)

mdoel = Unet()











