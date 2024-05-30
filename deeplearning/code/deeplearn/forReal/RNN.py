
import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l.torch import d2l

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split



# 加载数据并处理数据
FG_data = np.load("../data/DE_Normalization.npy")
FG_data = np.array(FG_data).astype(np.float32)
FG_data = torch.from_numpy(FG_data)
FG_data = torch.reshape(FG_data,(20355,5,17*16))  #RNN 通常期望输入数据的形状为 (batch_size, time_steps, features)
FG_labels = torch.from_numpy(np.load("../data/labels-numpy_data.npy"))
labels = torch.zeros([20355,1])
for i in range(20355):
    if FG_labels[i][0] > 0.35:
        labels[i] = 1



#划分训练集和验证集
dataset = TensorDataset(FG_data, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



# 定义模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  #添加一个回归层

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) #RNN 层输出的最后一个时间步的所有隐藏层单元的输出
        return out

input_size = 272  # 输入特征维度
hidden_size = 50  # 隐藏层大小
output_size = 1  # 输出大小 (二分类)

model = RNNModel(input_size, hidden_size, output_size)


# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
for epoch in range(num_epochs):
    model.train()
    correct = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        print(targets)
        predicted = (outputs > 0.5).float()
        correct += (predicted == targets).sum().item()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    accuracy = correct / len(train_loader.dataset)
    animator.add(epoch+1 ,(loss.item(), accuracy, None))
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # 验证模型
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    animator.add(epoch + 1, (None, None, accuracy))
    print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}')

plt.show()
