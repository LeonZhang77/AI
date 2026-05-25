import numpy as np
# 定义激活函数
def relu(x):
    # Relu
    # fx9x) = max(0,x)
    return np.maximum(0, x)

def sigmoid(x):
    # f(x) = 1/(1+e^(-x))
    # output arrange (0-1)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Main
def relu_derivative(x):
    return (x > 0).astype(float)

def predict(X,W1,b1,W2,b2,W3,b3):
    # 通过前向传播，来重新组合
    z1 = np.dot(X,W1)+b1
    a1 = relu(z1)
    z2 = np.dot(a1,W2)+b2
    a2 = relu(z2)
    z3 = np.dot(a2,W3)+b3
    y_pred = sigmoid(z3)
    return y_pred

if __name__ == '__main__':

# # 1. 输入三个条件，结果：是否旅游？
# # x1 = 1.0(天气好), x2 = 1.0(有空), x3 = 0.0(身体累)
#     X = np.array([[1.0, 1.0, 0.0]])
#
# # 2. 定义权重和偏置
# # 2.1 定义隐藏层1权重和偏置 (舒适性，必要性)
#     W1 = np.array([[0.8, 0.2],
#                    [0.1, 0.9],
#                    [0.5, 0.2]])
#     b1 = np.array([-0.2, 0.1])
#
# # 2.2 定义隐藏层2权重和偏置 (出门倾向, 风险顾虑)
#     W2 = np.array([[0.7, 0.7],
#                    [-0.4, 1.2]])
#     b2 = np.array([-0.1, 0.3])
#
# # 2.3 定义隐藏层3权重和偏置 (旅游概率)c
#     W3 = np.array([[0.8],
#                    [-0.5]])
#     b3 = np.array([-0.2])
#
#     target_loss = 0.000001
#     for epoch in range(1, 100000+1):
#     # 3 前向传播
#     # 3.1 隐藏层1： 线性求和 + 激活 (舒适性，必要性)
#         z1 = np.dot(X, W1) + b1 # 结果是1行2列
#         a1 = relu(z1)
#
#     # 3.2 隐藏层2： 线性求和 + 激活 (出门倾向, 风险顾虑)
#         z2 = np.dot(a1, W2) + b2
#         a2 = relu(z2)
#
#     # 3.3 输出层： 线性求和 + 激活
#         z3 = np.dot(a2, W3) + b3
#         y_pred = sigmoid(z3)
#
#         print('===============前向传播结果=======================')
#         print(f'输入 X: {X}')
#         print(f'隐藏层1求和: {z1}')
#         print(f'隐藏层1激活: {a1}')
#         print(f'隐藏层2求和: {z2}')
#         print(f'隐藏层2激活: {a2}')
#         print(f'输出层求和: {z3}')
#         print(f'预测值: {y_pred}')
#
#     # 4 计算损失值
#         y_true = 1
#     # 4.1 计算损失值 loss
#         loss = 0.5 * np.mean((y_true - y_pred) ** 2)
#         print(f'损失值loss: {loss:.6f}')
#
#     # 5 反向传播
#     # 5.1 输出层梯度
#     # 5.1.1 误差信号
#         detal_out = (y_pred - y_true)  * sigmoid_derivative(y_pred)
#     # 5.1.2 权重 + 偏置梯度 a2.T
#         dw3 = np.dot(a2.T, detal_out)
#         db3 = detal_out
#
#     # 5.2 隐藏层2梯度
#     # 5.2.1 误差信号 W3.T 权重矩阵倒置
#         detal2 = np.dot(detal_out, W3.T) * relu_derivative(z2)
#     # 5.2.2 权重 + 偏置梯度
#         dw2 = np.dot(a1.T, detal2)
#         db2 = detal2
#
#     # 5.3 隐藏层1梯度
#     # 5.3.1 误差信号 W3.T 权重矩阵倒置
#         detal1 = np.dot(detal2, W2.T) * relu_derivative(z2)
#     # 5.3.2 权重 + 偏置梯度
#         dw1 = np.dot(X.T, detal1)
#         db1 = detal1
#
#         print("===========返向传播结果===========")
#         print(f"输出层权重dw3梯度:{dw3}")
#         print(f"输出层偏置db3梯度:{db3}")
#         print(f"隐藏层2权重dw2梯度:{dw2}")
#         print(f"隐藏层2偏置db2梯度:{db3}")
#         print(f"隐藏层1权重dw1梯度:{dw1}")
#         print(f"隐藏层1偏置db1梯度:{db1}")
#
#     # 6 重新计算权重和偏置 学习率0。1
#     # 6.1 隐藏1权重和偏置
#         W1 = W1 - 0.1 * dw1
#         b1 = b1 - 0.1 * db1
#     # 6.2 隐藏2权重和偏置
#         W2 = W2 - 0.1 * dw2
#         b2 = b2 - 0.1 * db2
#     # 6.3 输出层权重和偏置
#         W3 = W3 - 0.1 * dw3
#         b3 = b3 - 0.1 * db3
#     # 7, 提前终止条件，损失少于预定值
#         if loss <= target_loss:
#             print(f'训练在 {epoch} 结束')
#             break
#
#     print(f"训练结束,共{epoch} 轮")
#     print(f"最终预测值,{y_pred},最终损失值:{loss:.8f}")
#
#     #8、保存MLP模型【神经网络】
#     model_file = "mpl_model.npz"
#     np.savez(model_file,
#              W1=W1,b1=b1,
#              W2=W2,b2=b2,
#              W3=W3,b3=b3)
#     print(f"模型已保存到：{model_file}")

    #8.1、读取模型
    model_file = "mpl_model.npz"
    data = np.load(model_file);
    W1_loaded = data['W1']
    b1_loaded = data['b1']
    W2_loaded = data['W2']
    b2_loaded = data['b2']
    W3_loaded = data['W3']
    b3_loaded = data['b3']

    #8.2、开始预测
    X = np.array([[0.5, 0.6, 0.5]])

    preds = predict(X, W1_loaded, b1_loaded,
                W2_loaded, b2_loaded,
                W3_loaded, b3_loaded)
    print(f"样本{X}--> 预测概率：{preds}")