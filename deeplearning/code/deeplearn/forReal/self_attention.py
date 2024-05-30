import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l.torch import d2l
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

# 加载数据并处理数据
FG_data = np.load("data/DE_Normalization.npy")
FG_data = np.array(FG_data).astype(np.float32)
FG_data = torch.from_numpy(FG_data)
FG_data = torch.reshape(FG_data,(20355,5,17,16))
FG_labels = torch.from_numpy(np.load("data/labels-numpy_data.npy"))
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

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # 计算 Query, Key, Value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        # 计算相似度分数
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # 计算加权值
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        # 融合输入和输出
        out = self.gamma * out + x
        return out

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.attention1 = SelfAttention(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.attention2 = SelfAttention(64)
        self.fc1 = nn.Linear(64 * 17 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # 二分类
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attention1(x)
        x = F.relu(self.conv2(x))
        x = self.attention2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型实例
model = AttentionModel()


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
