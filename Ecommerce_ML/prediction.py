
import numpy as np
import onnxruntime as ort
import pymysql
import json
from datetime import datetime

print('Hello World')

# ================= 1、MySQL连接配置 =================
DB_CONFIG = {
    'host': '192.168.31.233',
    'port': 3306,
    'user': 'root',
    'password': '123qwe!@#QWE',  # 根据实际环境修改
    'database': 'ecommerce',
    'charset': 'utf8mb4'
}
THRESHOLD = 0.7  # 触发营销的概率阈值

# ================= 2、加载ONNX模型 =================
MODEL_PATH = "purchase_intent_model.onnx"  # ONNX模型路径
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

#3、加载用户实时特征数据
def get_realtime_features(user_id):
    """从实时特征表查询最新一条特征"""
    conn = pymysql.connect(**DB_CONFIG)
    sql = """
            SELECT browse_cnt, cart_add_cnt, avg_stay_sec, favor_cnt,
                   prior_buy_game, age_normalized, member_level
            FROM user_realtime_features
            WHERE user_id = %s
            ORDER BY update_time DESC
            LIMIT 1
        """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id,))
        row = cur.fetchone()
    conn.close()
    #3.1、获取数据，并构造预测数据矩阵
    if row:
        return np.array([row], dtype=np.float32)
    else:
        return None

#6、存储预测结果
def save_prediction(user_id, features, prob, triggered):
    """将本次预测信息记录到prediction_log表"""
    conn = pymysql.connect(**DB_CONFIG)
    sql = """
            INSERT INTO prediction_log 
            (user_id, request_time, prob, features_json, threshold, triggered, coupon_action)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
    coupon = "满5000减300，享12期免息" if triggered else ""
    with conn.cursor() as cur:
        cur.execute(sql, (
            user_id,
            datetime.now(),  # 请求时间
            float(prob),  # 预测概率
            json.dumps(features.tolist()),  # 输入特征转为JSON
            THRESHOLD,  # 阈值
            int(triggered),  # 是否触发
            coupon
        ))
    conn.commit()
    conn.close()

def predict_and_act(user_id):
    # 4.1、获取用户预测特征数据
    features = get_realtime_features(user_id)
    # 4.2、ONNX Runtime 推理【预测】
    prob = session.run([output_name], {input_name: features})[0][0][0]
    triggered = prob > THRESHOLD
    print(f"用户 {user_id} | 购买概率: {prob:.4f} | 触发营销: {triggered}")
    # 4.3、存储预测结果
    save_prediction(user_id, features, prob, triggered)
    # 4.4、触发策略（实际项目中这里会调用营销系统API）
    if triggered:
        print(">>> 营销动作：弹出满5000减300优惠券，并显示'今日购买享12期免息'")
    else:
        print(">>> 暂不干预，继续浏览")

#5、调用预测函数实现预测
# ================= 测试入口 =================
if __name__ == "__main__":
    # 对几个用户依次进行预测
    test_users = ['u1', 'u2', 'u3', 'u4', 'u5']
    for uid in test_users:
        # 5.1、开始预测
        predict_and_act(uid)