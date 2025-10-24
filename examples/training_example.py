#!/usr/bin/env python3
"""
训练示例脚本
演示如何使用异常检测系统训练模型
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_detection.trainer import AnomalyDetectionTrainer
from src.anomaly_detection.data_loader import NetworkDataLoader
from shared.config import PROCESSED_DATA_DIR, MODEL_DIR
from shared.utils import setup_logging

def create_sample_data():
    """创建示例训练数据"""
    logger = setup_logging()
    logger.info("创建示例训练数据...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 生成正常流量数据
    normal_data = []
    for i in range(1000):
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
        normal_data.append(flow)
    
    # 生成DDoS攻击数据
    ddos_data = []
    for i in range(200):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'dst_ip': "10.0.0.1",  # 固定目标
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([80, 443]),
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(1),  # 短时间
            'total_packets': np.random.poisson(1000),  # 大量包
            'total_bytes': np.random.poisson(100000),  # 大量字节
            'avg_packet_size': np.random.normal(100, 20),  # 小包
            'packets_per_second': np.random.normal(100, 20),  # 高包率
            'bytes_per_second': np.random.normal(10000, 2000),  # 高字节率
            'min_packet_size': np.random.normal(40, 5),
            'max_packet_size': np.random.normal(200, 50),
            'std_packet_size': np.random.normal(50, 10),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 3600),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.01, 0.005),  # 短间隔
            'inter_packet_time_std': np.random.normal(0.005, 0.002),
            'src_to_dst_packets': np.random.poisson(500),
            'dst_to_src_packets': np.random.poisson(500),
            'src_to_dst_bytes': np.random.poisson(50000),
            'dst_to_src_bytes': np.random.poisson(50000),
            'tcp_flags': 2,  # SYN标志
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0,
            'anomaly_type': 'ddos'
        }
        ddos_data.append(flow)
    
    # 生成端口扫描数据
    port_scan_data = []
    for i in range(150):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 10)}",  # 少量源IP
            'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.randint(1, 65535),  # 随机目标端口
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(0.1),  # 极短时间
            'total_packets': np.random.poisson(1),  # 少量包
            'total_bytes': np.random.poisson(100),  # 少量字节
            'avg_packet_size': np.random.normal(60, 10),
            'packets_per_second': np.random.normal(10, 5),
            'bytes_per_second': np.random.normal(600, 100),
            'min_packet_size': np.random.normal(40, 5),
            'max_packet_size': np.random.normal(80, 10),
            'std_packet_size': np.random.normal(10, 5),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 1800),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.1, 0.05),
            'inter_packet_time_std': np.random.normal(0.05, 0.02),
            'src_to_dst_packets': np.random.poisson(1),
            'dst_to_src_packets': 0,  # 无响应
            'src_to_dst_bytes': np.random.poisson(60),
            'dst_to_src_bytes': 0,
            'tcp_flags': 2,  # SYN标志
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0,
            'anomaly_type': 'port_scan'
        }
        port_scan_data.append(flow)
    
    # 合并所有数据
    all_data = normal_data + ddos_data + port_scan_data
    df = pd.DataFrame(all_data)
    
    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存数据
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    data_path = os.path.join(PROCESSED_DATA_DIR, "sample_training_data.parquet")
    df.to_parquet(data_path, index=False)
    
    logger.info(f"示例数据已保存到: {data_path}")
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"类别分布:\n{df['anomaly_type'].value_counts()}")
    
    return data_path

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始训练示例...")
    
    try:
        # 1. 创建示例数据
        data_path = create_sample_data()
        
        # 2. 创建训练器
        trainer = AnomalyDetectionTrainer()
        
        # 3. 执行训练
        logger.info("开始训练模型...")
        results = trainer.full_training_pipeline(
            data_path=data_path,
            target_column='anomaly_type'
        )
        
        # 4. 输出结果
        logger.info("训练完成!")
        logger.info(f"模型保存路径: {results['model_path']}")
        logger.info(f"测试集准确率: {results['test_results']['metrics']['accuracy']:.4f}")
        logger.info(f"F1分数: {results['test_results']['metrics']['f1_macro']:.4f}")
        
        # 5. 显示特征重要性
        importance = results['test_results']['feature_importance']
        logger.info("Top 10 特征重要性:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            logger.info(f"{i+1:2d}. {feature}: {score:.4f}")
        
    except Exception as e:
        logger.error(f"训练示例失败: {e}")
        raise

if __name__ == "__main__":
    main()
