import numpy as np
import torch
from torch import nn
# 初始化init
# 访问和初始化每层的参数
# 1
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
w1 = net1[0].weight
b1 = net1[0].bias
net1[0].weight.data = torch.Tensor(np.random.uniform(3, 5, size=(40, 30)))

for layer in net1:
    if isinstance(layer, nn.Linear): # 判断是否是线性层
        param_shape = layer.weight.shape
        layer.weight.data = torch.Tensor(np.random.normal(0, 0.5, size=param_shape))

# 2
class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )

        self.l1[0].weight.data = torch.randn(40, 30)  # 直接对某一层初始化

        self.l2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

net2 = sim_net()
# children 只会访问到模型定义中的第一层，而 modules 会访问到最后的结构
for i in net2.children():
    print(i)
for i in net2.modules():
    print(i)


for layer in net2.modules():
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.Tensor(np.random.normal(0, 0.5, size=param_shape))

# 内置init  初始化
nn.init.xavier_uniform_(net1[0].weight)

