import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten,Linear,Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from C_10model import *

#设置数据集
train_data = torchvision.datasets.CIFAR10("../data",train=True,transform=torchvision.transforms.ToTensor(),
                                       download=True)
test_data = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)

train_dataLoader = DataLoader(train_data,batch_size=64)
test_dataLoader = DataLoader(train_data,batch_size=64)

#设置网络模型
g_mod = G_mod()
loss_fn = nn.CrossEntropyLoss()
learn_rate = 0.01
optim = torch.optim.SGD(g_mod.parameters(),lr=learn_rate)

total_train_step = 0
total_test_step = 0

epoch = 10

writer = SummaryWriter("../logs")

for i in range(epoch):
    print("---------第{}轮训练开始-----------".format(i))
    #训练
    g_mod.train()
    for data in train_dataLoader:
        imgs,targets =data
        output = g_mod(imgs)
        loss = loss_fn(output,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step = total_train_step + 1;
        if total_train_step % 100 == 0:
            print("训练次数： {}。Loss： {}".format(total_train_step,loss.item()))
            writer.add_scalar("train loss ",loss.item(),total_train_step)
    #验证
    total_test_loss = 0
    total_accurary = 0
    g_mod.eval()
    with torch.no_grad:
        for data in test_dataLoader:
            imgs, targets = data
            output = g_mod(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_step + loss.item()
            accurary = (output.argmax(1)==targets).sum()
            total_accurary = total_accurary + accurary.item()

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accurary/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_loss", total_accurary/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

writer.close()




