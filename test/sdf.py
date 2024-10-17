import torch
import torch.nn as nn

# 假设有3个样本，每个样本有4个类别
outputs = torch.randn(3, 4)  # 输出预测张量
labels = torch.tensor([1, 2, 3])  # 标签张量，需要转换为one-hot编码形式
criterion = nn.CrossEntropyLoss()  # 创建多维交叉熵损失对象
loss = criterion(outputs, labels)  # 计算损失值
print(outputs, labels, loss, sep="\n")
