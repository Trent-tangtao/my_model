import torch
from torch.autograd import Variable
import numpy as np

print(torch.__version__)
# 一元线性回归    回归问题

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

w = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)
x_train= Variable(x_train)
y_train= Variable(y_train)

def line_model(x):
    return w*x+b

def get_loss(y_,y):
    return torch.mean((y_-y)**2)

y_ = line_model(x_train)
loss = get_loss(y_, y_train)
loss.backward()

lr = 1e-4

for e in range(20):
    y_ = line_model(x_train)
    loss = get_loss(y_, y_train)
    w.grad.zero_() #记得梯度归零
    b.grad.zero_()
    loss.backward()
    w.data = w.data-lr*w.grad.data
    b.data = b.data-lr*b.grad.data
    print('epoch:{},loss:{}'.format(e,loss))


# Logistic Sigmoid   分类问题
import torch.functional as F
def logistic_regression(x):
    return F.sigmoid(torch.mm(x,w)+b)

# 优化器
from torch import nn
# loss=nn.MSELoss()
loss = nn.BCEWithLogitsLoss()
w = nn.Parameter(torch.randn(2,1))
b = nn.Parameter(torch.zeros(1))
optimizer = torch.optim.SGD([w,b],lr=0.01)

# 三步走
optimizer.zero_grad()
loss.backward()
optimizer.step()


# Sequential   Module
seq_net=nn.Sequential(
    nn.Linear(2,4),
    nn.Tanh(),
    nn.Linear(4,1)
)
print(seq_net[0]) #第一层
w0= seq_net[0].weight
param =seq_net.parameters()

# 同时保存模型和参数
torch.save(seq_net,"save.pth")
seq_net1 = torch.load("save.pth")
# 只保存参数
torch.save(seq_net.state_dict(),"save_params.pth")
seq_net2=nn.Sequential(
    nn.Linear(2,4),
    nn.Tanh(),
    nn.Linear(4,1)
)
seq_net2.load_state_dict(torch.load("save_params.pth"))


# Module
class net(nn.Module):
    def __init__(self,num_input,num_hidden,num_output):
        super(net, self).__init__()
        self.layer1=nn.Linear(num_input,num_hidden)
        self.layer2=nn.Tanh()
        self.layer3=nn.Linear(num_hidden,num_output)
    def forward(self, x):
         x = self.layer1(x)
         x = self.layer2(x)
         x = self.layer3(x)
         return x

net = net(2,4,1)
l1=net.layer1
print(l1,l1.weight)


class net2(nn.Module):
    def __init__(self,num_input,num_hidden,num_output):
        super(net2, self).__init__()
        self.network=nn.Sequential(
            nn.Linear(num_input,num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden,num_input)
        )
    def forward(self, x):
         x = self.network(x)
         return x

net = net2(2,4,1)
print(net.network)