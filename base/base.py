import torch
import numpy as np

# Tensor基本性质，numpy
numpy_tensor = np.random.randn(10,20)
torch_tensor = torch.Tensor(numpy_tensor)
# print(torch_tensor.size())
# print(torch_tensor.shape)
# print(torch_tensor.dim())
# print(torch_tensor.type())
# print(torch_tensor.numel())

tensor1= torch.randn(3,4)
tensor1= tensor1.type(torch.DoubleTensor)
tensor1 = tensor1.float()
# print(tensor1)
# print(type(tensor1.numpy()))
# print(tensor1.numpy().dtype)

x = torch.ones(2,3)
x = x.long()
# print(x.type())
x = torch.randn(2,3)
y = torch.randn(2,3)
#  每行的最大值
max_value,max_index= torch.max(x, dim=1)
sum_x = torch.sum(x,dim =1)
# print(max_value, max_index,sum_x)
z = x+y
# z = torch.add(x,y)


# 增加或者减少维度
x = x.unsqueeze(0)  # 第一维增加
x = x.unsqueeze(1)  # 第二位增加
x = x.squeeze(0)  # 减少第一维
x = x.squeeze()   # 去掉所有的一维
# print(x.shape)

# 维度转换
x = torch.randn(3,4,5)
x = x.permute(1,0,2)  # 重新排列维度
x = x.transpose(0,1)  # 交换两个维度
# print(x.shape)

# reshape
x = x.view(-1, 5) #  -1表示任意大小，5表示第二维变成5
x = x.view(2, 30)  # reshape,注意总元素不变
# print(x.shape)


#### pytorch支持inplace操作，直接对tensor操作，不需要另外开辟空间
#### 一般都是在操作符后面加_
x.unsqueeze_(0)
x.unsqueeze(1)
x.squeeze_()   # 总元素也不能减少！
# print(x.shape)



# Variable对tensor的封装， .data  .grad  .grad_fn什么方式得到的

from torch.autograd import Variable
x_tensor=torch.randn(10,5)
y_tensor=torch.randn(10,5)
x=Variable(x_tensor,requires_grad=True)
y=Variable(y_tensor,requires_grad=True)
z = torch.sum(x+y)
# print(z.data)
# print(z.grad_fn)
z.backward()
# print(x.grad)
# print(y.grad)

# 尝试构建一个函数 y = x^2 ，然后求 x=2 的导数
x = Variable(torch.FloatTensor([2]), requires_grad=True)
y = x**2
y.backward()
# print(y.grad_fn)
# print(x.grad)
# 注意，是求某个点的倒数


# 自动求导

x = Variable(torch.Tensor([2]),requires_grad=True)
y = x+2
z = y**2
# print(z.grad_fn)
z.backward()
# print(x.grad)

# 自动求导一次后，计算图就会被丢弃   保留后，  每次计算的梯度会相加,记得归零
# z.backward(retain_graph=True)

# 明确导数的意义，没看懂，等实战

# 静态图和动态图

