import numpy as np
import torch
import torch.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch import nn


def vgg_block(num_convs,in_channels,out_channels):
    net=[nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU(True)]
    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        net.append(nn.ReLU(True))
    net.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*net)

block_demo=vgg_block(3,64,128)
# print(block_demo)
input_demo = Variable(torch.zeros(1,64,300,300))
output_demo = block_demo(input_demo)
# print(output_demo.shape)

def vgg_stack(num_convs, channels):
    net=[]
    for n,c in zip(num_convs,channels):
        in_c=c[0]
        out_c=c[1]
        net.append(vgg_block(n,in_c,out_c))
    return nn.Sequential(*net)

vgg_net =vgg_stack((1,1,2,2,2),((3,64),(64,128), (128, 256), (256, 512), (512, 512)))

class vgg(nn.Module):
    def __init__(self):
        super(vgg,self).__init__()
        self.feature=vgg_net
        self.fc=nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)   # 拉平
        x = self.fc(x)
        return x




from torch.utils.data import DataLoader


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.Tensor(x)
    return x


train_set = CIFAR10('./data', train=True, transform=data_tf,download=True)
train_data = DataLoader(train_set, batch_size=4, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=data_tf,download=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

net = vgg()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()


from torch.utils.trainer import Trainer
from torch.utils.trainer.plugins import AccuracyMonitor, Logger

T = Trainer(model=net, optimizer=optimizer, criterion=criterion, dataset=train_data)
m = AccuracyMonitor()
l = Logger([m.stat_name, 'accuracy.last'])
T.register_plugin(m)
T.register_plugin(l)
T.run(epochs=2)
print(T.stats)
