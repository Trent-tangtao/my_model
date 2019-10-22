import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.functional as F

# Conv => BN => 激活 => pooling
### 卷积 池化   输入(batch, channel, H, W)
#  nn.Conv2d() 相当于直接定义了一层卷积网络结构，而使用 torch.nn.functional.conv2d() 相当于定义了一个卷积的操作
#  nn.MaxPool2d()，一种是 torch.nn.functional.max_pool2d()
# 激活 nn.ReLU(True)
#  inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出。

### BN  torch.nn.BatchNorm1d() 和 torch.nn.BatchNorm2d()
# 批标准化,对于每一层网络的输出，对其做一个归一化，使其服从标准的正态分布
# 测试的时候该使用批标准化吗?
# 答案是肯定的，因为训练的时候使用了，而测试的时候不使用肯定会导致结果出现偏差，但是测试的时候如果只有一个数据集，那么均值不就是这个值，方差为 0 吗？
#     这显然是随机的，所以测试的时候不能用测试的数据集去算均值和方差，而是用训练的时候算出的移动平均均值和方差去代替

### 现在的网络往往不用dropout，而是用正则化，L1，L2
# 可以看到加完正则项之后会对参数做更大程度的更新 ==> 这也被称为权重衰减(weight decay)
# torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)
# 注意正则项的系数的大小非常重要，如果太大，会极大的抑制参数的更新，导致欠拟合，如果太小，那么正则项这个部分基本没有贡献，
# 所以选择一个合适的权重衰减系数非常重要，这个需要根据具体的情况去尝试，初步尝试可以使用 1e-4 或者 1e-3


### 学习率衰减
# 学习率衰减之前应该经过充分的训练，比如训练 80 次或者 100 次，然后再做学习率衰减得到更好的结果，有的时候甚至需要做多次学习率衰减
# optimizer.param_groups[0]['lr'] = 1e-5
# def set_learning_rate(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
# ...
# if epoch == 80:
#         set_learning_rate(optimizer, 0.01)

### 自带模型与数据集
from torchvision.models import VGG
from torchvision.datasets import mnist

### 数据增强
from PIL import Image
from torchvision import transforms as tfs
im=Image.open("../test.png")
# 1.对图片进行一定比例缩放
new_im=tfs.Resize((100, 200))(im)
# 2.对图片进行随机位置的截取
random_im=tfs.RandomCrop((100, 200))(im)
center_im=tfs.CenterCrop((100, 200))(im)
# 3.对图片进行随机的水平和竖直翻转
h_flip=tfs.RandomHorizontalFlip()(im)  #水平
v_filp=tfs.RandomVerticalFlip()(im) #垂直
# 4.对图片进行随机角度的旋转
rot_im=tfs.RandomRotation(45)(im)
# rot_im.show()
# 5.对图片进行亮度、对比度和颜色的随机变化
# 亮度： 随机从 0 ~ 2 之间亮度变化，1 表示原图
bright_im = tfs.ColorJitter(brightness=1)(im)
# 对比度：随机从 0 ~ 2 之间对比度变化，1 表示原图
contrast_im = tfs.ColorJitter(contrast=1)(im)
# 颜色： 随机从 -0.5 ~ 0.5 之间对颜色变化
color_im = tfs.ColorJitter(hue=0.5)(im)
#联合起来compose
im_aug = tfs.Compose([
    tfs.Resize(120),
    tfs.RandomHorizontalFlip(),
    tfs.RandomCrop(96),
    tfs.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)
])

import matplotlib.pyplot as plt
nrows = 3
ncols = 3
figsize = (8, 8)
_, figs = plt.subplots(nrows, ncols, figsize=figsize)
for i in range(nrows):
    for j in range(ncols):
        figs[i][j].imshow(im_aug(im))
        figs[i][j].axes.get_xaxis().set_visible(False)
        figs[i][j].axes.get_yaxis().set_visible(False)
plt.show()


def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(120),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(96),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

def test_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(96),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x



