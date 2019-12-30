import tensorflow as tf
import os
import numpy as np
from model import Unet
from data_preprocess import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

img_save_pth = "/home/zhangzhichao/CNN/wangxi/train/images_gen"
mask_save_pth = "/home/zhangzhichao/CNN/wangxi/train/masks_gen"

dataset = Dataset(img_save_pth, mask_save_pth)

x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='input')
y = tf.placeholder(tf.int32, shape=[None, 256, 256], name='ground-truth')

model  = Unet()
logits = model.model(x)

y_ = tf.nn.softmax(logits, name='logits_prob')
predict = tf.argmax(y_, axis=3, name='predicts')

print(y_.get_shape().as_list())
print(predict.get_shape().as_list())
print(y.get_shape().as_list())

# 在使用 tf.nn.sparse_softmax_cross_entropy_with_logits() 时需要注意参数logits的输入是modeld的最后一层未经过激活曾的输出，否则回报错
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
# tensorboard 可视化
tf.summary.scalar("loss", loss)

train_step = tf.train.AdamOptimizer().minimize(loss)

# save model
saver = tf.train.Saver()

merge = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir="./log", graph=sess.graph)
    for epoch in range(500):
        for i in range(625):
            image, mask = dataset.next_batch(16)
            summary, acc, _ = sess.run([merge, loss, train_step], feed_dict={x: image, y: mask})
            print("epoch: % 05d, batch: %d, loss: %.5f" % (epoch, i, acc))
            summary_writer.add_summary(summary, (epoch % 500) + i)

        if epoch % 100 == 0:
            saver.save(sess, "./checkpoints/model_%04d.ckpt" % (epoch))




# img_save_pth = r"C:\Users\AIR\Desktop\xldownload\dataset\train\images_gen"
# mask_save_pth = r"C:\Users\AIR\Desktop\xldownload\dataset\train\masks_gen"
#
# x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='input')
# y = tf.placeholder(tf.float32, shape=[None, 256, 256], name='ground-truth')
#
# model  = Unet()
# logits = model.model(x)
# y_ = tf.nn.softmax(logits)
# y_ = tf.argmax(y_, axis=-1)
#
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_))
# train_step = tf.train.AdamOptimizer().minimize(loss)
#
# dataset = Dataset(img_save_pth, mask_save_pth)
#
# with tf.Session as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epoch in range(6000):
#         for i in range(625):
#             image, mask = dataset.next_batch(16)
#             aloss, _ = sess.run([loss, train_step], feed_dict={x: image, y: mask})
#
#             print("epoch: % 05d, batch: %d, acc: %.5f" % (epoch, i, acc ))
#
#
#
#
#
