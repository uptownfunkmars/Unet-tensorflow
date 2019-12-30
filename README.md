# Unet-tensorflow

使用tensorflow对Unet的复现, 切割图片部分单纯切割未使用过采样以及镜像填充。
使用的数据集为CCF卫星影像的AI分类与识别提供的数据集初赛复赛训练集，一共五张卫星遥感影像。
（地址参考https://github.com/ximimiao/deeplabv3-Tensorflow）

data_preprocess.py 用作切图以及数据类，主要用于在训练时像model中提供batch。
model.py           为tensorflow复现模型。
train.py           训练过程。
