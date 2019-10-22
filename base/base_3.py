import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import torch.functional as F

# 28*28拉平
def data_tf(x):
    # 0 是白色， 255 是黑色  归一化
    x = np.array(x,dtype='float32') / 255
    x = (x-0.5)/0.5
    x = x.reshape((-1,))
    x = torch.Tensor(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf,download=True)
test_set=mnist.MNIST('./data',train=False, transform=data_tf,download=True)
a_data, a_label = train_set[0]
# print(a_data.shape)

# 数据迭代器   数据太大，无法一次读入内存
from torch.utils.data import DataLoader
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

a, a_label = next(iter(train_data))
print(a.shape)
print(a_label.shape)

class net(nn.Module):
    def __init__(self,input=784,output=10):
        super(net, self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input,400),
            nn.ReLU(),
            nn.Linear(400,200),
            nn.ReLU(),
            nn.Linear(200,output)
        )
    def forward(self, x):
        x = self.network(x)
        return x

net1 = net()
criterion= nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net1.parameters(),1e-1)

losses=[]
acces=[]
eval_loss=[]
eval_acc=[]
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss=0
    train_acc=0
    net1.train()
    for im,label in train_data:
        im=Variable(im)
        label=Variable(label)
        out=net1(im)
        loss=criterion(out,label)
        optimizer.zero_grad()
        optimizer.step()
        train_loss = loss.item()
        _, pre =out.max(1)
        num_crrect = (pre==label).sum().item()
        acc=num_crrect/im.shape[0]
        train_acc+=acc
    losses.append(train_loss/len(train_data))
    acces.append(train_acc/len(train_data))

    eval_loss = 0
    eval_acc = 0
    net1.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net1(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print("epoch{},train_loss{},train_acc{}".format(e,train_loss/len(train_data),train_acc/len(train_data)))

