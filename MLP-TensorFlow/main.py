import numpy as np
#导入 TensorFlow
import tensorflow as tf
#导入 keras
import keras
from keras.models import Sequential,Model
from keras.layers import Input, Dense
from keras.optimizers import SGD,Adam
# 导入tf2onnx库，用于将tensorflow 模型转换为onnx格式
import tf2onnx
# 导入onnx库，保存和处理 onnx模型文件
import onnx
# 导入onnx Runtime 库 用于加载 onnx模型 进行推理验证
import onnxruntime as ort


def set_model(X_train, y_train):
    # 2 定义MLP模型
    # 优化：神经元个数=特征数量 3 + 输出 1 + 冗余 = 4 ~ 6 个
    # model = Sequential([
    #     Dense(4, activation='relu', input_shape=(3,), name='hidden1'), #隐藏层1
    #     # Dense(2, activation='relu', input_shape=(2,), name='hidden2'), #隐藏层2
    #     Dense(1, activation='sigmoid', input_shape=(1,), name='output'), #输出层
    #     ])

    # 2.1、定义输入层
    inputs = Input(shape=(3,), name='input_features')
    # 2.2、定义隐藏层
    x = Dense(4, activation='relu', name='hidden_layer')(inputs)
    # 2.3、定义输出层
    outputs = Dense(1, activation='sigmoid', name='output_layer')(x)
    # 2.4、创建模型
    model = Model(inputs=inputs, outputs=outputs)

    # 3 定义编译模型
    # 优化： 损失函数 = 均分方差 / 二无交叉熵
    # 定义损失率
    # model.compile(optimizer=SGD(learning_rate=0.1),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy']) # 跟踪损失值
    model.compile(optimizer=Adam(0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])  # 跟踪损失值
    return model

def train_model(model, X_train, y_train):
    # 4 训练模型
    model.fit(X_train, y_train, epochs=5000, verbose=1, validation_split=0.2)

    # 5 计算损失值
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f'训练集准确率: {train_loss: .4f}')

    # 准备新样本，评估模型
    X_new = np.array([[0.0, 0.0, 0.0]])
    pred = model.predict(X_new)
    print(f"tensorflow [0, 0, 0]预测概率:{pred[0][0]:.6f}")

    X_new1 = np.array([[1.0, 1.0, 0.0]])
    pred1 = model.predict(X_new1)
    print(f"tensorflow [1, 1, 0]预测概率:{pred1[0][0]:.6f}")

    X_new2 = np.array([[0.0, 1.0, 1.0]])
    pred2 = model.predict(X_new2)
    print(f"tensorflow [0, 1, 1]预测概率:{pred2[0][0]:.6f}")
    return

def save_onnx_model(model, onnx_model_path):
    # ---------------------保存为ONNX----------------------
    # opset=13 指定ONNX算子集版本 确认兼容性
    # 1、将tensorflow 模型转换为onnx 模型
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

    # 2、保存onnx_model
    onnx.save(onnx_model, onnx_model_path)
    print(f"onnx 模型 已保存为:{onnx_model_path}")
    return

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print("Hello World")
    # 1. 输入三个条件，结果：是否旅游？
    # x1 = 1.0(天气好), x2 = 1.0(有空), x3 = 0.0(身体累)
    X_train = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1], 
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    # 真实标签 1.0 表示去旅游
    y_train = np.array([[1], [0], [0], [0], [0], [0], [0], [0]], dtype=np.float32)

    # model = set_model(X_train, y_train)
    #
    # train_model(model, X_train, y_train)
    #
    # save_onnx_model(model, "train_model.onnx")

    # ---------------------使用ONNX模型预测----------------------
    # 1、加载
    ort_session = ort.InferenceSession("train_model.onnx")
    # 2、获取 输入层和输出层
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    # 3、预测
    X_new3 = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    onnx_result = ort_session.run([output_name], {input_name: X_new3})[0]
    print(f"onnx模型预测概率[1.0, 1.0, 0]:{onnx_result[0][0]:.6f}")

    X_new4 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    onnx_result1 = ort_session.run([output_name], {input_name: X_new4})[0]
    print(f"onnx模型预测概率[0.0, 1.0, 0.0]:{onnx_result1[0][0]:.6f}")


