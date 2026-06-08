-- =============================================
-- 创建电商数据库
-- =============================================
CREATE DATABASE IF NOT EXISTS ecommerce CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ecommerce;

-- =============================================
-- 1. 用户行为流水表
-- 记录每一次页面浏览、加购、收藏、下单事件
-- =============================================
DROP TABLE IF EXISTS user_behavior_log;
CREATE TABLE user_behavior_log (
    log_id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '日志ID',
    user_id VARCHAR(32) NOT NULL COMMENT '用户ID',
    session_id VARCHAR(64) NOT NULL COMMENT '会话ID',
    event_type VARCHAR(20) NOT NULL COMMENT '事件类型: browse/cart_add/favor/order',
    product_id VARCHAR(32) COMMENT '商品ID',
    event_time DATETIME(3) NOT NULL COMMENT '事件发生时间(毫秒精度)',
    stay_duration_sec INT DEFAULT 0 COMMENT '页面停留秒数(仅browse事件有值)',
    INDEX idx_user_time (user_id, event_time) COMMENT '按用户和时间查询',
    INDEX idx_session_time (session_id, event_time) COMMENT '按会话和时间查询'
) ENGINE=InnoDB;

-- =============================================
-- 2. 用户画像表
-- 存储变化较慢的用户属性
-- =============================================
DROP TABLE IF EXISTS user_profile;
CREATE TABLE user_profile (
    user_id VARCHAR(32) PRIMARY KEY COMMENT '用户ID',
    age_normalized FLOAT COMMENT '归一化年龄(0~1)',
    member_level TINYINT COMMENT '会员等级 1-5',
    prior_buy_game TINYINT COMMENT '历史是否购买游戏类商品 0/1',
    reg_date DATE COMMENT '注册日期'
) ENGINE=InnoDB;

-- =============================================
-- 3. 训练样本表
-- 由行为日志聚合生成，供模型离线训练
-- =============================================
DROP TABLE IF EXISTS training_samples;
CREATE TABLE training_samples (
    sample_id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '样本ID',
    user_id VARCHAR(32) COMMENT '用户ID',
    sample_time DATETIME COMMENT '样本截止时间',
    browse_cnt INT COMMENT '近5分钟浏览次数',
    cart_add_cnt INT COMMENT '近5分钟加购次数',
    avg_stay_sec FLOAT COMMENT '近5分钟平均停留秒数',
    favor_cnt INT COMMENT '近5分钟收藏次数',
    prior_buy_game TINYINT COMMENT '历史是否购买游戏',
    age_normalized FLOAT COMMENT '归一化年龄',
    member_level TINYINT COMMENT '会员等级',
    label TINYINT COMMENT '未来5分钟是否购买 1/0',
    INDEX idx_sample_time (sample_time) COMMENT '按样本时间查询'
) ENGINE=InnoDB;

-- =============================================
-- 4. 预测日志表
-- 记录线上每次推理的结果，用于监控和迭代
-- =============================================
DROP TABLE IF EXISTS prediction_log;
CREATE TABLE prediction_log (
    pred_id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '预测ID',
    user_id VARCHAR(32) COMMENT '用户ID',
    request_time DATETIME(3) COMMENT '预测请求时间',
    prob FLOAT COMMENT '模型输出的购买概率',
    features_json TEXT COMMENT '输入特征JSON',
    threshold FLOAT COMMENT '本次使用的阈值',
    triggered TINYINT COMMENT '是否触发营销策略 1/0',
    coupon_action VARCHAR(100) COMMENT '触发的优惠券内容'
) ENGINE=InnoDB;

-- =============================================
-- 5. 实时用户特征表
-- 线上服务查询此表获取最新聚合特征
-- 生产环境中由Flink等流处理引擎实时更新
-- =============================================
DROP TABLE IF EXISTS user_realtime_features;
CREATE TABLE user_realtime_features (
    user_id VARCHAR(32) COMMENT '用户ID',
    session_id VARCHAR(64) COMMENT '会话ID',
    browse_cnt INT COMMENT '近5分钟浏览次数',
    cart_add_cnt INT COMMENT '近5分钟加购次数',
    avg_stay_sec FLOAT COMMENT '近5分钟平均停留秒数',
    favor_cnt INT COMMENT '近5分钟收藏次数',
    prior_buy_game TINYINT COMMENT '历史购买游戏',
    age_normalized FLOAT COMMENT '归一化年龄',
    member_level TINYINT COMMENT '会员等级',
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '特征更新时间',
    PRIMARY KEY (user_id, session_id)
) ENGINE=InnoDB;

-- =============================================
-- 插入模拟数据，用于演示流程
-- =============================================

-- 用户画像
INSERT INTO user_profile VALUES
('u1',0.25,4,1,'2025-01-15'),
('u2',0.15,2,0,'2025-03-10'),
('u3',0.40,5,1,'2024-11-20'),
('u4',0.10,1,0,'2025-05-01'),
('u5',0.32,4,0,'2025-02-28');

-- 用户1的行为序列（购买场景）
INSERT INTO user_behavior_log (user_id,session_id,event_type,product_id,event_time,stay_duration_sec) VALUES
('u1','s1','browse','p1','2026-05-30 10:00:00.000',150),
('u1','s1','cart_add','p1','2026-05-30 10:01:00.000',0),
('u1','s1','browse','p2','2026-05-30 10:02:00.000',120),
('u1','s1','favor','p1','2026-05-30 10:03:00.000',0),
('u1','s1','order','p1','2026-05-30 10:04:00.000',0);

-- 用户2的行为序列（未购买）
INSERT INTO user_behavior_log (user_id,session_id,event_type,product_id,event_time,stay_duration_sec) VALUES
('u2','s2','browse','p3','2026-05-30 10:00:00.000',20),
('u2','s2','browse','p4','2026-05-30 10:01:00.000',25),
('u2','s2','browse','p5','2026-05-30 10:03:00.000',15);

-- 用户3的行为序列（购买场景）
INSERT INTO user_behavior_log (user_id,session_id,event_type,product_id,event_time,stay_duration_sec) VALUES
('u3','s3','browse','p6','2026-05-30 10:00:00.000',300),
('u3','s3','cart_add','p6','2026-05-30 10:02:00.000',0),
('u3','s3','favor','p6','2026-05-30 10:03:00.000',0),
('u3','s3','order','p6','2026-05-30 10:04:30.000',0);

-- 用户4的行为序列（未购买）
INSERT INTO user_behavior_log (user_id,session_id,event_type,product_id,event_time,stay_duration_sec) VALUES
('u4','s4','browse','p7','2026-05-30 10:00:00.000',10);

-- 用户5的行为序列（未购买，用于测试）
INSERT INTO user_behavior_log (user_id,session_id,event_type,product_id,event_time,stay_duration_sec) VALUES
('u5','s5','browse','p8','2026-05-30 10:00:00.000',100);

-- 预填充实时特征表（与上述行为日志中的聚合结果一致）
INSERT INTO user_realtime_features VALUES
('u1','s1',2,1,135.0,1,1,0.25,4,NOW()),
('u2','s2',3,0,20.0,0,0,0.15,2,NOW()),
('u3','s3',1,1,300.0,1,1,0.40,5,NOW()),
('u4','s4',1,0,10.0,0,0,0.10,1,NOW()),
('u5','s5',1,0,100.0,0,0,0.32,4,NOW());