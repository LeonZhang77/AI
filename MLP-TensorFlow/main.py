
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print("Hello World")
    # 1. 输入三个条件，结果：是否旅游？
    # x1 = 1.0(天气好), x2 = 1.0(有空), x3 = 0.0(身体累)
    X_train = np.array([[1.0, 1.0, 0.0]])
    # 真实标签 1.0 表示去旅游
    y_train = np.array([1.0])

    # 2 定义MLP模型
    model = Sequential([
        Dense(2, activation='relu', input_shape=(3,), name='hidden1'), #隐藏层1
        Dense(2, activation='relu', input_shape=(2,), name='hidden2'), #隐藏层2
        Dense(1, activation='sigmoid', input_shape=(1,), name='output'), #输出层
        ])

    # 3 定义编译模型
    model.compile(optimizer=SGD(learning_rate=0.01),
                  loss='mean_squared_error')

    # 4 训练模型
    history = model.fit(X_train, y_train, epochs=1000, verbose=1)

    # 5 计算损失值
    print(f'损失值loss: {history.history["loss"][-1]}')

    # 准备新样本，评估模型
    X_new = np.array([[0.0, 0.0, 0.0]])
    y_pred = model.predict(X_new)
    print(f'预测值: {y_pred[0][0]}')