#!/usr/bin/env python3
"""
细粒度异常分类功能测试脚本
验证模型是否能区分：暴力破解、欺骗、上传危机、数据库攻击
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.utils import setup_logging

def create_fine_grained_test_data():
    """创建细粒度攻击分类测试数据"""
    logger = setup_logging()
    logger.info("创建细粒度攻击分类测试数据...")
    
    np.random.seed(42)
    test_data = []
    
    # 1. 正常流量 (20个样本)
    for i in range(20):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(100, 200)}",
            'dst_ip': f"10.0.0.{np.random.randint(1, 10)}",
            'src_port': np.random.randint(49152, 65535),
            'dst_port': np.random.choice([80, 443, 22]),
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(5),
            'total_packets': np.random.poisson(20),
            'total_bytes': np.random.poisson(2000),
            'avg_packet_size': np.random.normal(1000, 200),
            'packets_per_second': np.random.normal(4, 1),
            'bytes_per_second': np.random.normal(4000, 1000),
            'attack_type': 'normal'
        }
        test_data.append(flow)
    
    # 2. 暴力破解攻击 (20个样本) - 大量失败登录尝试
    for i in range(20):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 50)}",
            'dst_ip': f"10.0.0.{np.random.randint(1, 5)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': 22,  # SSH端口
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(0.1),  # 短时间大量尝试
            'total_packets': np.random.poisson(100),      # 大量包
            'total_bytes': np.random.poisson(6000),       # 大量字节
            'avg_packet_size': np.random.normal(60, 10),  # 小包
            'packets_per_second': np.random.normal(1000, 200),  # 高频率
            'bytes_per_second': np.random.normal(60000, 10000),
            'attack_type': 'brute_force'
        }
        test_data.append(flow)
    
    # 3. 欺骗攻击 (20个样本) - IP欺骗
    for i in range(20):
        flow = {
            'src_ip': f"10.0.0.{np.random.randint(1, 10)}",  # 内网IP
            'dst_ip': f"192.168.1.{np.random.randint(1, 50)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([80, 443, 53]),
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(0.05),
            'total_packets': np.random.poisson(5),
            'total_bytes': np.random.poisson(300),
            'avg_packet_size': np.random.normal(60, 10),
            'packets_per_second': np.random.normal(100, 50),
            'bytes_per_second': np.random.normal(6000, 2000),
            'attack_type': 'spoofing'
        }
        test_data.append(flow)
    
    # 4. 上传危机攻击 (20个样本) - 恶意文件上传
    for i in range(20):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 100)}",
            'dst_ip': f"10.0.0.{np.random.randint(1, 10)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': 80,  # HTTP端口
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(2),
            'total_packets': np.random.poisson(50),
            'total_bytes': np.random.poisson(100000),     # 大文件上传
            'avg_packet_size': np.random.normal(2000, 500),  # 大包
            'packets_per_second': np.random.normal(25, 10),
            'bytes_per_second': np.random.normal(50000, 20000),
            'attack_type': 'upload_attack'
        }
        test_data.append(flow)
    
    # 5. 数据库攻击 (20个样本) - SQL注入等
    for i in range(20):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 100)}",
            'dst_ip': f"10.0.0.{np.random.randint(1, 10)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': 3306,  # MySQL端口
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(1),
            'total_packets': np.random.poisson(30),
            'total_bytes': np.random.poisson(15000),
            'avg_packet_size': np.random.normal(500, 100),
            'packets_per_second': np.random.normal(30, 10),
            'bytes_per_second': np.random.normal(15000, 5000),
            'attack_type': 'database_attack'
        }
        test_data.append(flow)
    
    # 创建DataFrame
    df = pd.DataFrame(test_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"细粒度测试数据创建完成，形状: {df.shape}")
    logger.info("攻击类型分布:")
    print(df['attack_type'].value_counts())
    
    return df

def test_fine_grained_classification():
    """测试细粒度异常分类功能"""
    logger = setup_logging()
    logger.info("=== 测试细粒度异常分类功能 ===")
    
    try:
        # 1. 创建测试数据
        test_df = create_fine_grained_test_data()
        
        # 2. 检查当前模型是否支持细粒度分类
        logger.info("检查当前模型配置...")
        
        # 检查配置文件中的攻击类型
        from shared.config import Labels
        current_types = Labels.ANOMALY_TYPES
        logger.info(f"当前模型支持的攻击类型: {list(current_types.keys())}")
        
        # 需要的攻击类型
        required_types = ['normal', 'brute_force', 'spoofing', 'upload_attack', 'database_attack']
        logger.info(f"需要的攻击类型: {required_types}")
        
        # 3. 检查数据集
        logger.info("检查训练数据集...")
        train_df = pd.read_excel('data/train.xlsx')
        logger.info(f"训练数据标签: {train_df['label'].unique()}")
        logger.info("训练数据只有二分类标签，无法支持细粒度分类")
        
        # 4. 分析问题
        logger.info("=== 问题分析 ===")
        missing_types = set(required_types) - set(current_types.keys())
        if missing_types:
            logger.warning(f"缺少的攻击类型: {missing_types}")
        
        # 5. 生成测试报告
        test_report = {
            'test_timestamp': datetime.now().isoformat(),
            'test_purpose': '细粒度异常分类功能验证',
            'required_attack_types': required_types,
            'current_model_types': list(current_types.keys()),
            'missing_types': list(missing_types),
            'dataset_limitation': '训练数据只有二分类标签(0/1)',
            'conclusion': '当前模型不具备细粒度异常分类功能',
            'recommendations': [
                '需要获取包含细粒度攻击标签的训练数据',
                '需要修改模型配置支持5种攻击类型',
                '需要重新训练模型',
                '需要验证细粒度分类性能'
            ]
        }
        
        # 保存测试报告
        report_path = os.path.join('data/processed', 'fine_grained_classification_test_report.json')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试报告已保存到: {report_path}")
        
        return test_report
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始细粒度异常分类功能测试...")
    
    try:
        test_report = test_fine_grained_classification()
        
        logger.info("=== 测试结论 ===")
        logger.info("❌ 当前模型不具备细粒度异常分类功能")
        logger.info("📋 需要改进:")
        for rec in test_report['recommendations']:
            logger.info(f"  - {rec}")
            
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        raise

if __name__ == "__main__":
    main()
