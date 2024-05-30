import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from FG_model import *

FG_data = np.load("../data/DE_Normalization.npy")
FG_data = np.array(FG_data).astype(np.float32)
FG_data = torch.from_numpy(FG_data)
FG_data = torch.reshape(FG_data,(20355,5,17,16))



FG_labels = torch.from_numpy(np.load("../data/labels-numpy_data.npy"))
labels = torch.zeros([20355,2])
for i in range(20355):
    if FG_labels[i][0] < 0.35:
        labels[i][0] = 1
    else:
        labels[i][1] = 1

print(labels.shape)
model = G_mod()
model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

learn_rate = 0.001
optim = torch.optim.SGD(model.parameters(), lr=learn_rate)


writer = SummaryWriter("../logs")

total_train_step = 0
total_test_step = 0
for i in range(3):
    kfold = KFold(n_splits=10)
    for train_idx, val_idx in enumerate(kfold.split(FG_data)):
        print("---------第{}轮第{}折训练开始-----------".format(i + 1,train_idx + 1))
        # 训练
        model.train()
        for idx in val_idx[0]:
            input = FG_data[idx].float()
            target = labels[idx].float()
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            #output = torch.reshape(output, [1])
            loss = loss_fn(output[0], target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_train_step = total_train_step + 1
            if total_train_step % 2000 == 0:
                print("训练次数： {}。Loss： {}".format(total_train_step,loss.item()))
                writer.add_scalar("train loss ",loss.item(),total_train_step)
        # 验证
        print("---------第{}轮第{}折验证开始-----------".format(i+1,train_idx + 1))
        total_test_loss = 0
        total_accurary = 0
        model.eval()
        total_test_num = 0
        with torch.no_grad():
            for idx in val_idx[1]:
                input = FG_data[idx].float()
                target = labels[idx].float()
                input = input.cuda()
                target = target.cuda()

                output = model(input)
                loss = loss_fn(output[0], target)
                total_test_loss = total_test_loss + loss.item()
                accurary = output[0].argmax().item() == target.argmax().item()

                total_accurary = total_accurary + accurary
                total_test_num = total_test_num + 1

        print("整体测试集上的loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accurary / total_test_num))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accurary", total_accurary / total_test_num, total_test_step)
        total_test_step = total_test_step + 1
writer.close()
