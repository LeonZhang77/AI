import tensorflow as tf
import numpy as np

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print('Hello World!')

    # ==================== 1. 定义输入图像（4×4，单通道）====================
    # 与手算例子相同的输入
    X = np.array([
        [0.2, 0.5, 0.1, 0.8],
        [0.3, 0.9, 0.6, 0.2],
        [0.7, 0.4, 0.5, 0.3],
        [0.1, 0.0, 0.9, 0.4]
    ], dtype=np.float32)
    # TensorFlow 需要 batch 维度 + 通道维度 → shape (1, 4, 4, 1)
    X_input = X.reshape(1, 4, 4, 1)

    # ==================== 2. 构建模型（Sequential）====================
    model = tf.keras.Sequential([
        # 卷积层：1个 2x2 卷积核，输入通道=1，输出通道=1，无填充，步长=1
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=2,
            strides=1,
            padding='valid',
            use_bias=True,
            activation=None,          # 先不加激活，手动加ReLU
            input_shape=(4, 4, 1)
        ),
        tf.keras.layers.ReLU(),       # ReLU 激活
        # 池化层：2x2 最大池化，步长=1（允许重叠）
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='valid'),
        tf.keras.layers.Flatten(),
        # 全连接层：输出2个单元，无激活（logits）
        tf.keras.layers.Dense(2, activation=None)
    ])

    # ==================== 3. 手动设置卷积核、偏置和全连接层权重 ====================
    # 卷积核：手算例子中的 2x2 矩阵，需要 reshape 为 (height, width, in_channels, out_channels)
    # 原核 K = [[1.0, 0.5], [0.0, -0.5]]
    conv_kernel = np.array([[[[1.0]], [[0.5]]], [[[0.0]], [[-0.5]]]], dtype=np.float32)
    # 或者更直观的构建方式：
    # conv_kernel = np.array([1.0, 0.5, 0.0, -0.5], dtype=np.float32).reshape(2,2,1,1)
    conv_bias = np.array([0.1], dtype=np.float32)   # 偏置

    # 全连接层权重：2x4 矩阵，偏置 [0.1, 0.2]
    fc_weights = np.array([
        [0.5, 0.2, 0.1, 0.3],
        [0.1, 0.4, 0.2, 0.6]
    ], dtype=np.float32)
    fc_bias = np.array([0.1, 0.2], dtype=np.float32)

    # 获取模型层
    conv_layer = model.layers[0]          # Conv2D
    fc_layer = model.layers[3]            # Dense (Flatten 是第2层? 索引: 0 Conv2D, 1 ReLU, 2 MaxPooling, 3 Flatten, 4 Dense)
    # 修正索引：实际顺序：0 Conv2D, 1 ReLU, 2 MaxPooling2D, 3 Flatten, 4 Dense
    fc_layer = model.layers[4]

    # 设置权重
    conv_layer.set_weights([conv_kernel, conv_bias])
    fc_layer.set_weights([fc_weights.T, fc_bias])

    # 可选：打印验证权重是否设置成功
    print("卷积核已设置:", conv_layer.get_weights()[0].squeeze())
    print("卷积偏置:", conv_layer.get_weights()[1])
    print("全连接权重:", fc_layer.get_weights()[0])
    print("全连接偏置:", fc_layer.get_weights()[1])

    # ==================== 4. 前向传播 ====================
    # 方式1：使用 model.predict
    logits = model.predict(X_input)   # shape (1,2)
    logits = logits[0]                # 去掉 batch 维

    # 计算 Softmax 概率
    probs = tf.nn.softmax(logits).numpy()

    # 手算预期结果：logits = [1.255, 1.565], probs ≈ [0.4237, 0.5763]
    print("\n前向传播结果：")
    print(f"Logits: {logits}")
    print(f"Softmax 概率: {probs}")
    print(f"预测类别: {np.argmax(probs)}")
