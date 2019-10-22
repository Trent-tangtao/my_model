import torch
import torch.nn as nn
import torch.optim as opt
from torchvision import datasets, transforms
from torch.utils.trainer import Trainer
from torch.utils.trainer.plugins import AccuracyMonitor, Logger

model = yourModel(num_classes=10)
print(model)
optimizer = opt.SGD(filter(lambda p: p.requires_grad,model.parameters()),lr = 0.001)
criterion = nn.CrossEntropyLoss()

normalize = transforms.Normalize(mean = [0.491, 0.482, 0.447],std = [0.247, 0.243, 0.262])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/data/torchvison', train=True, download=True,
                     transform = transforms.Compose([
                         transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),
                         normalize])), batch_size = 16, shuffle = True,
)
dataset = train_loader.dataset
dataset.train_data = dataset.train_data[:32]
validate_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/data/torchvison', train=False,
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])),
batch_size = 1000, shuffle = False,
)

T = Trainer(model = model,optimizer = optimizer,criterion = criterion,dataset = train_loader)
m = AccuracyMonitor()
l = Logger([m.stat_name, 'accuracy.last'])
T.register_plugin(m)
T.register_plugin(l)
T.run(epochs=1)
print(T.stats)