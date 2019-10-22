import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tfs
from torchvision import models

net= models.resnet50(pretrained=True)
# print(net)
for param in net.parameters():
    param.requires_grad = False
# 可以直接改网络！！
net.fc=nn.Linear(2048,2)
# 可以选择优化哪部分的参数
optimizer = torch.optim.SGD(net.fc.parameters(),lr=1e-2,weight_decay=1e-4)