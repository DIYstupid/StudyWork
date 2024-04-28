import torch

# 定义模型参数
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
print("w = ", w)
print("b = ", b)
# 定义输入和目标输出
x = torch.tensor([2.0])
y_true = torch.tensor([4.0])

# 定义损失函数
loss_fn = torch.nn.MSELoss()

# 定义优化器
optimizer = torch.optim.SGD([w, b], lr=0.1)

# 迭代训练
for i in range(100):
    # 前向传播
    y_pred = w * x + b
    loss = loss_fn(y_pred, y_true)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新模型参数
    optimizer.step()

# 输出模型参数
print("w = ", w)
print("b = ", b)
