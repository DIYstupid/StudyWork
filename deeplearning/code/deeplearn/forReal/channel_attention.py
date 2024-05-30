import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l.torch import d2l
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

# 加载数据并处理数据
FG_data = np.load("../data/DE_Normalization.npy")
FG_data = np.array(FG_data).astype(np.float32)
FG_data = torch.from_numpy(FG_data)
FG_data = torch.reshape(FG_data,(20355,5,17,16))
FG_labels = torch.from_numpy(np.load("../data/labels-numpy_data.npy"))
labels = torch.zeros([20355])
for i in range(20355):
    if FG_labels[i][0] > 0.35:
        labels[i] = 1
labels = torch.tensor(labels, dtype=torch.long)


#划分训练集和验证集
dataset = TensorDataset(FG_data, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        #将每个通道的特征图池化为单个值，即每个通道的平均值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #原始输入张量与权重相乘，以调整每个通道的特征图，从而增强重要通道
        return x * y.expand_as(x)

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.att1 = ChannelAttention(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.att2 = ChannelAttention(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 自适应平均池化层，确保固定输出尺寸
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.att1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)

        x = self.conv2(x)
        x = self.att2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建模型实例
model = CNNWithAttention()


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    running_loss = running_loss / len(train_loader)
    accuracy = correct / total
    animator.add(epoch+1 ,(running_loss, accuracy, None))
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}")

    # 测试模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in val_loader:

            data, labels = data.to(device), labels.to(device).float()
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = correct / total

        animator.add(epoch + 1, (None, None, test_accuracy))
        print(f"Accuracy: {test_accuracy:.2f}")

plt.show()
