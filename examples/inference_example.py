#!/usr/bin/env python3
"""
推理示例脚本
演示如何使用训练好的模型进行异常检测
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_detection.inferencer import TwoStageAnomalyDetector
from src.anomaly_detection.data_loader import NetworkDataLoader
from shared.config import MODEL_DIR, PROCESSED_DATA_DIR
from shared.utils import setup_logging

def create_test_data():
    """创建测试数据"""
    logger = setup_logging()
    logger.info("创建测试数据...")
    
    # 设置随机种子
    np.random.seed(123)
    
    # 生成测试数据（包含正常和异常流量）
    test_data = []
    
    # 正常流量
    for i in range(50):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([80, 443, 22, 21, 25, 53]),
            'protocol': np.random.choice(['TCP', 'UDP']),
            'flow_duration': np.random.exponential(10),
            'total_packets': np.random.poisson(50),
            'total_bytes': np.random.poisson(5000),
            'avg_packet_size': np.random.normal(1000, 200),
            'packets_per_second': np.random.normal(5, 2),
            'bytes_per_second': np.random.normal(5000, 2000),
            'min_packet_size': np.random.normal(64, 10),
            'max_packet_size': np.random.normal(1500, 100),
            'std_packet_size': np.random.normal(200, 50),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 86400),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.1, 0.05),
            'inter_packet_time_std': np.random.normal(0.02, 0.01),
            'src_to_dst_packets': np.random.poisson(25),
            'dst_to_src_packets': np.random.poisson(25),
            'src_to_dst_bytes': np.random.poisson(2500),
            'dst_to_src_bytes': np.random.poisson(2500),
            'tcp_flags': np.random.randint(0, 256),
            'udp_length': np.random.randint(8, 1500),
            'icmp_type': np.random.randint(0, 16),
            'icmp_code': np.random.randint(0, 16),
            'anomaly_type': 'normal'
        }
        test_data.append(flow)
    
    # DDoS攻击流量
    for i in range(20):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'dst_ip': "10.0.0.1",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': 80,
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(0.5),
            'total_packets': np.random.poisson(2000),
            'total_bytes': np.random.poisson(200000),
            'avg_packet_size': np.random.normal(100, 20),
            'packets_per_second': np.random.normal(200, 50),
            'bytes_per_second': np.random.normal(20000, 5000),
            'min_packet_size': np.random.normal(40, 5),
            'max_packet_size': np.random.normal(200, 50),
            'std_packet_size': np.random.normal(50, 10),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 1800),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.005, 0.002),
            'inter_packet_time_std': np.random.normal(0.002, 0.001),
            'src_to_dst_packets': np.random.poisson(1000),
            'dst_to_src_packets': np.random.poisson(1000),
            'src_to_dst_bytes': np.random.poisson(100000),
            'dst_to_src_bytes': np.random.poisson(100000),
            'tcp_flags': 2,
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0,
            'anomaly_type': 'ddos'
        }
        test_data.append(flow)
    
    # 端口扫描流量
    for i in range(15):
        flow = {
            'src_ip': "192.168.1.100",
            'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.randint(1, 65535),
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(0.05),
            'total_packets': np.random.poisson(1),
            'total_bytes': np.random.poisson(60),
            'avg_packet_size': np.random.normal(60, 10),
            'packets_per_second': np.random.normal(20, 10),
            'bytes_per_second': np.random.normal(1200, 200),
            'min_packet_size': np.random.normal(40, 5),
            'max_packet_size': np.random.normal(80, 10),
            'std_packet_size': np.random.normal(10, 5),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 900),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.05, 0.02),
            'inter_packet_time_std': np.random.normal(0.02, 0.01),
            'src_to_dst_packets': np.random.poisson(1),
            'dst_to_src_packets': 0,
            'src_to_dst_bytes': np.random.poisson(60),
            'dst_to_src_bytes': 0,
            'tcp_flags': 2,
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0,
            'anomaly_type': 'port_scan'
        }
        test_data.append(flow)
    
    # 创建DataFrame
    df = pd.DataFrame(test_data)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # 保存测试数据
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    test_path = os.path.join(PROCESSED_DATA_DIR, "test_data.parquet")
    df.to_parquet(test_path, index=False)
    
    logger.info(f"测试数据已保存到: {test_path}")
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"真实类别分布:\n{df['anomaly_type'].value_counts()}")
    
    return test_path

def find_latest_model():
    """查找最新的模型文件"""
    logger = setup_logging()
    
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"模型目录不存在: {MODEL_DIR}")
    
    model_files = []
    for file in os.listdir(MODEL_DIR):
        if file.startswith("xgboost_anomaly_model_") and file.endswith(".pkl"):
            model_files.append(os.path.join(MODEL_DIR, file))
    
    if not model_files:
        raise FileNotFoundError("未找到XGBoost模型文件")
    
    # 返回最新的模型
    latest_model = max(model_files, key=os.path.getctime)
    logger.info(f"使用模型: {latest_model}")
    
    return latest_model

def single_prediction_example():
    """单样本预测示例"""
    logger = setup_logging()
    logger.info("=== 单样本预测示例 ===")
    
    try:
        # 查找模型
        model_path = find_latest_model()
        
        # 创建检测器
        detector = TwoStageAnomalyDetector(model_path)
        
        # 创建单个样本
        sample_data = {
            'src_ip': '192.168.1.100',
            'dst_ip': '10.0.0.1',
            'src_port': 12345,
            'dst_port': 80,
            'protocol': 'TCP',
            'flow_duration': 0.1,
            'total_packets': 1000,
            'total_bytes': 100000,
            'avg_packet_size': 100,
            'packets_per_second': 100,
            'bytes_per_second': 10000,
            'min_packet_size': 40,
            'max_packet_size': 200,
            'std_packet_size': 50,
            'flow_start_time': datetime.now().timestamp() - 100,
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': 0.001,
            'inter_packet_time_std': 0.0005,
            'src_to_dst_packets': 500,
            'dst_to_src_packets': 500,
            'src_to_dst_bytes': 50000,
            'dst_to_src_bytes': 50000,
            'tcp_flags': 2,
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0
        }
        
        # 转换为numpy数组
        feature_columns = detector.xgboost_model.feature_columns
        X = np.array([list(sample_data.values())[:len(feature_columns)]]).reshape(1, -1)
        
        # 执行预测
        result = detector.predict_single(X[0])
        
        logger.info(f"预测结果: {result['label']}")
        logger.info(f"置信度: {result['confidence']:.4f}")
        logger.info(f"检测路径: {result['detection_path']}")
        logger.info(f"各类别概率: {result['probabilities']}")
        
    except Exception as e:
        logger.error(f"单样本预测失败: {e}")

def batch_prediction_example():
    """批量预测示例"""
    logger = setup_logging()
    logger.info("=== 批量预测示例 ===")
    
    try:
        # 创建测试数据
        test_path = create_test_data()
        
        # 查找模型
        model_path = find_latest_model()
        
        # 创建检测器和数据加载器
        detector = TwoStageAnomalyDetector(model_path)
        data_loader = NetworkDataLoader()
        
        # 加载和预处理数据
        df = data_loader.load_from_file(test_path)
        df_processed = data_loader.preprocess_network_data(df)
        
        # 准备特征
        feature_columns = detector.xgboost_model.feature_columns
        X = df_processed[feature_columns].values
        
        # 执行批量预测
        results = detector.batch_predict(X, confidence_threshold=0.5)
        
        # 分析结果
        logger.info(f"总样本数: {len(df)}")
        logger.info(f"预测异常数: {sum(1 for label in results['labels'] if label != 'normal')}")
        logger.info(f"高置信度预测: {sum(1 for conf in results['confidences'] if conf > 0.8)}")
        logger.info(f"需要人工审核: {sum(1 for label in results['labels'] if label == 'needs_review')}")
        
        # 显示预测分布
        from collections import Counter
        prediction_counts = Counter(results['labels'])
        logger.info("预测结果分布:")
        for label, count in prediction_counts.items():
            logger.info(f"  {label}: {count}")
        
        # 计算准确率（如果有真实标签）
        if 'anomaly_type' in df.columns:
            true_labels = df['anomaly_type'].tolist()
            correct = sum(1 for true, pred in zip(true_labels, results['labels']) 
                         if true == pred or (true == 'normal' and pred == 'normal'))
            accuracy = correct / len(true_labels)
            logger.info(f"准确率: {accuracy:.4f}")
        
        # 保存结果
        result_df = df_processed.copy()
        result_df['predicted_label'] = results['labels']
        result_df['confidence'] = results['confidences']
        result_df['detection_path'] = results['detection_paths']
        
        output_path = os.path.join(PROCESSED_DATA_DIR, "prediction_results.parquet")
        result_df.to_parquet(output_path, index=False)
        logger.info(f"预测结果已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"批量预测失败: {e}")

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始推理示例...")
    
    try:
        # 1. 单样本预测示例
        single_prediction_example()
        
        # 2. 批量预测示例
        batch_prediction_example()
        
        logger.info("推理示例完成!")
        
    except Exception as e:
        logger.error(f"推理示例失败: {e}")
        raise

if __name__ == "__main__":
    main()
