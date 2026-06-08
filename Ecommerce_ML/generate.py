
import numpy as np
import onnxruntime as ort
import pymysql
import json
from datetime import datetime


print('PyCharm')

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

def generate_samples():
    """
    从原始行为日志生成训练样本：
    按用户和会话聚合近5分钟行为，并打标签（未来5分钟内是否有下单）
    """
    conn = pymysql.connect(**DB_CONFIG)
    sql = """
    INSERT INTO training_samples 
    (user_id, sample_time, browse_cnt, cart_add_cnt, avg_stay_sec, favor_cnt,
     prior_buy_game, age_normalized, member_level, label)
    SELECT
        ubl.user_id,
        MAX(ubl.event_time) AS sample_time,
        SUM(CASE WHEN event_type = 'browse' THEN 1 ELSE 0 END) AS browse_cnt,
        SUM(CASE WHEN event_type = 'cart_add' THEN 1 ELSE 0 END) AS cart_add_cnt,
        AVG(CASE WHEN event_type = 'browse' THEN stay_duration_sec END) AS avg_stay_sec,
        SUM(CASE WHEN event_type = 'favor' THEN 1 ELSE 0 END) AS favor_cnt,
        up.prior_buy_game,
        up.age_normalized,
        up.member_level,
        -- 标签：该会话首次事件后5分钟内是否有下单
        CASE WHEN EXISTS(
            SELECT 1 FROM user_behavior_log ubl2 
            WHERE ubl2.user_id = ubl.user_id
              AND ubl2.session_id = ubl.session_id
              AND ubl2.event_type = 'order'
              AND ubl2.event_time BETWEEN MIN(ubl.event_time) 
                  AND DATE_ADD(MIN(ubl.event_time), INTERVAL 5 MINUTE)
        ) THEN 1 ELSE 0 END AS label
    FROM user_behavior_log ubl
    JOIN user_profile up ON ubl.user_id = up.user_id
    GROUP BY ubl.user_id, ubl.session_id
    HAVING sample_time IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
    conn.close()
    print("训练样本生成完毕")

#5、调用预测函数实现预测
# ================= 测试入口 =================
if __name__ == "__main__":
    generate_samples()