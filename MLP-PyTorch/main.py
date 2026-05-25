import numpy as np
import torch
# 神经网络模块
import torch.nn as nn
# 优化器
import torch.optim as optim

# 定义MLP神经网络基础模型
class TravelMlp(nn.Module):
    def __init__(self):
        super(TravelMlp, self).__init__()
        # 定义隐藏层1
        self.hidden1 = nn.Linear(3, 2) # 3个输入条件，2个神经元
        # 定义隐藏层2
        self.hidden2 = nn.Linear(2, 2)
        # 定义输出层
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        # 定义隐藏层1激活函数
        x = torch.relu(self.hidden1(x))
        # 定义隐藏层2激活函数
        x = torch.relu(self.hidden2(x))
        # 定义输出层激活函数
        x = torch.sigmoid(self.output(x))
        return x

if __name__ == '__main__':
    # 1. 输入三个条件，结果：是否旅游？
    # x1 = 1.0(天气好), x2 = 1.0(有空), x3 = 0.0(身体累)
    X_train = torch.tensor([[1.0, 1.0, 0.0]])
    # 真实标签 1.0 表示去旅游
    y_train = torch.tensor([1.0])

    # 2 定义隐藏层1/隐藏层2/输出层
    model = TravelMlp()

    # 3 定义损失函数
    criterion = nn.MSELoss() #均分方差

    # 4 定义学习率，优化器，偏置，权重
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 5 训练
    epochs = 100000
    for epoch in range(epochs):
        y_pred = model(X_train)
        # 计算损失值
        loss = criterion(y_pred, y_train)
        print(f'训练: {epoch} 次')
        print(f'损失值loss: {loss.item():.8f}')
        # 实现反向传播
        optimizer.zero_grad() # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step() # 更新参数

    # 训练结束
    print(f'训练完成')

    # 准备新样本，评估模型
    model.eval()
    X_new = torch.tensor([[0.0, 0.0, 0.0]])
    y_pred_new = model(X_new)
    loss = criterion(y_pred_new, y_train)
    print(f'预测值: {y_pred_new.item():.8f}')
    print(f'损失值loss: {loss.item():.8f}')