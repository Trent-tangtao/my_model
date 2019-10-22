import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# ResNet是跨层求和 ，enseNet是跨层channel拼接
# 因为是在通道维度进行特征的拼接，所以底层的输出会保留进入所有后面的层，这能够更好的保证梯度的传播，
# 同时能够使用低维的特征和高维的特征进行联合训练，能够得到更好的结果。

#卷积块的顺序是 bn -> relu -> conv
def conv_block(in_channel, out_channel):
    layer=nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel,out_channel,3,stride=1,padding=1,bias=False)
    )
    return layer

#dense block 将每次的卷积的输出称为 growth_rate，因为如果输入是 in_channel，
# 有 n 层，那么输出就是 in_channel + n * growh_rate
class dense_block(nn.Module):
    def __init__(self,in_channel,growth_rate,num_layers):
        super(dense_block,self).__init__()
        block=[]
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel,growth_rate))
            channel+=growth_rate
        self.net=nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out=layer(x)
            x=torch.cat((out,x),dim=1)
        return x

#过渡层（transition block），因为 DenseNet 会不断地对维度进行拼接， 所以当层数很高的时候，输出的通道数就会越来越大，
# 参数和计算量也会越来越大，为了避免这个问题，需要引入过渡层将输出通道降低下来，同时也将输入的长宽减半，这个过渡层可以使用 1 x 1 的卷积
def transition(in_channel,out_channel):
    trans_layer=nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel,out_channel,1),
        nn.AvgPool2d(2,2)
    )
    return trans_layer

class densenet(nn.Module):
    def __init__(self,in_channel,num_classes, growth_rate=32,block_layers=[6, 12, 24, 16]):
        super(densenet,self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channel,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3,2,padding=1)
        )
        channels=64
        block=[]
        for i,layers in enumerate(block_layers):
            block.append(dense_block(channels,growth_rate,layers))
            channels+=layers*growth_rate
            if i!=len(block_layers)-1:
                block.append(transition(channels,channels//2))
                channels=channels//2
        self.block2=nn.Sequential(*block)
        self.block2.add_module('bn',nn.BatchNorm2d(channels))
        self.block2.add_module('relu',nn.ReLU(True))
        self.block2.add_module('avg_pool',nn.AvgPool2d(3))

        self.classifier=nn.Linear(channels,num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    test_net = densenet(3, 10)
    test_x = Variable(torch.zeros(1, 3, 96, 96))
    test_y = test_net(test_x)
    print('output: {}'.format(test_y.shape))




