import numpy as np
import pymysql
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import tf2onnx
import onnx
import warnings
# ================= 1、MySQL连接配置 =================
DB_CONFIG = {
    'host': '192.168.31.233',
    'port': 3306,
    'user': 'root',
    'password': '123qwe!@#QWE',  # 根据实际环境修改
    'database': 'ecommerce',
    'charset': 'utf8mb4'
}
MODEL_PATH = "purchase_intent_model.onnx"  # ONNX模型路径
THRESHOLD = 0.7  # 触发营销的概率阈值

def load_training_data():
    """从 training_samples 表读取全部特征和标签"""
    conn = pymysql.connect(**DB_CONFIG)
    sql = """
        SELECT browse_cnt, cart_add_cnt, avg_stay_sec, favor_cnt,
               prior_buy_game, age_normalized, member_level, label
        FROM training_samples
        ORDER BY sample_id
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        raise RuntimeError("训练样本表为空，请先执行样本生成")

    # 前7列为特征，最后一列为标签
    X = np.array([row[:-1] for row in rows], dtype=np.float32)
    y = np.array([[row[-1]] for row in rows], dtype=np.float32)
    return X, y

#5、调用预测函数实现预测
# ================= 测试入口 =================
if __name__ == "__main__":
    X_train, y_train = load_training_data()
    print(f"训练样本数量: {X_train.shape[0]}")

    # 2. 构建MLP神经网络
    inputs = Input(shape=(7,), name='user_features')
    x = Dense(16, activation='relu', name='hidden1')(inputs)
    x = Dense(8, activation='relu', name='hidden2')(x)
    outputs = Dense(1, activation='sigmoid', name='purchase_prob')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # 3. 训练模型
    model.fit(
        X_train, y_train,
        epochs=2000,
        batch_size=4,
        validation_split=0.2,  # 20%作为验证集
        verbose=1
    )

    # 4. 评估训练集准确率
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"训练集准确率: {train_acc:.4f}")

    # 5.模型保存
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
    onnx_path = "purchase_intent_model.onnx"
    onnx.save(onnx_model, onnx_path)
    print(f"ONNX模型已保存至: {onnx_path}")