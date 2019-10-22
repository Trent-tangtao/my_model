import numpy as np
import torch
from torch import nn
import torch.functional as F

class Trainer():
    def __init__(self,model,criterion,optimizer,dataset,USE_CUDA):
        self.model=model
        self.criterion=criterion
        self.optimizer=optimizer
        self.dataset=dataset
        self.iterations=0
        self.USE_CUDA=USE_CUDA

    def run(self,epochs=1):
        for i in range(1, epochs+1):
            self.train()

    def train(self):
        for i,data in enumerate(self.dataset,self.iterations+1):
            batch_input, batch_target = data
            input_var = batch_input
            target_var = batch_target
            if self.USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()

            # 每一次前馈就是一次函数闭包操作
            def closure():
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                return loss

            # loss 返回,准备优化
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
        self.iterations += i


