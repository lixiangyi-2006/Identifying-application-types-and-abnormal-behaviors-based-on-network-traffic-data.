#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集整合脚本
整合五个数据集：良性、暴力破解、欺骗、上传危机、数据库攻击
并按合理比例划分训练集和测试集
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_label_dataset(file_path, label_name, sample_size=None):
    """加载数据集并添加标签"""
    logger.info(f"加载数据集: {file_path}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        logger.info(f"原始数据形状: {df.shape}")
        
        # 如果指定了采样大小，进行采样
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"采样后数据形状: {df.shape}")
        
        # 添加标签列
        df['anomaly_type'] = label_name
        
        logger.info(f"添加标签 '{label_name}' 完成")
        return df
        
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

def merge_datasets():
    """整合所有数据集"""
    logger.info("开始整合数据集...")
    
    # 数据集路径和标签映射
    datasets = {
        "D:/data1/良性数据集.xlsx": "normal",
        "D:/data1/暴力破解（猜解密码等攻击）的数据集.xlsx": "brute_force", 
        "D:/data1/欺骗（IP来源伪造等）的数据集.xlsx": "spoofing",
        "D:/data1/上传危机的数据集.xlsx": "upload_attack",
        "D:/data1/数据库攻击的数据集.xlsx": "database_attack"
    }
    
    all_datasets = []
    
    # 加载每个数据集
    for file_path, label in datasets.items():
        if os.path.exists(file_path):
            df = load_and_label_dataset(file_path, label)
            all_datasets.append(df)
        else:
            logger.warning(f"文件不存在: {file_path}")
    
    if not all_datasets:
        raise ValueError("没有找到任何有效的数据集文件")
    
    # 合并所有数据集
    logger.info("合并所有数据集...")
    merged_df = pd.concat(all_datasets, ignore_index=True)
    
    logger.info(f"合并后总数据形状: {merged_df.shape}")
    logger.info("标签分布:")
    print(merged_df['anomaly_type'].value_counts())
    
    return merged_df

def split_train_test(merged_df, test_ratio=0.2, stratify=True):
    """划分训练集和测试集"""
    logger.info(f"开始划分训练集和测试集 (测试集比例: {test_ratio})")
    
    if stratify:
        # 分层抽样，保持各类别比例
        train_df, test_df = train_test_split(
            merged_df, 
            test_size=test_ratio, 
            random_state=42, 
            stratify=merged_df['anomaly_type']
        )
    else:
        # 随机划分
        train_df, test_df = train_test_split(
            merged_df, 
            test_size=test_ratio, 
            random_state=42
        )
    
    logger.info(f"训练集形状: {train_df.shape}")
    logger.info(f"测试集形状: {test_df.shape}")
    
    logger.info("训练集标签分布:")
    print(train_df['anomaly_type'].value_counts())
    
    logger.info("测试集标签分布:")
    print(test_df['anomaly_type'].value_counts())
    
    return train_df, test_df

def analyze_dataset_balance(df, dataset_name):
    """分析数据集平衡性"""
    logger.info(f"=== {dataset_name} 平衡性分析 ===")
    
    label_counts = df['anomaly_type'].value_counts()
    total_samples = len(df)
    
    print(f"总样本数: {total_samples:,}")
    print("各类别分布:")
    for label, count in label_counts.items():
        percentage = count / total_samples * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")
    
    # 计算平衡性指标
    min_count = label_counts.min()
    max_count = label_counts.max()
    balance_ratio = min_count / max_count
    
    print(f"平衡比例: {balance_ratio:.3f}")
    if balance_ratio > 0.5:
        print("数据集相对平衡")
    elif balance_ratio > 0.2:
        print("数据集轻度不平衡")
    else:
        print("数据集严重不平衡")
    
    return balance_ratio

def save_datasets(train_df, test_df, output_dir="data/processed"):
    """保存训练集和测试集"""
    logger.info("保存数据集...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_path = os.path.join(output_dir, "train.xlsx")
    train_df.to_excel(train_path, index=False)
    logger.info(f"训练集已保存到: {train_path}")
    
    # 保存测试集
    test_path = os.path.join(output_dir, "test.xlsx")
    test_df.to_excel(test_path, index=False)
    logger.info(f"测试集已保存到: {test_path}")
    
    # 保存数据统计信息
    stats = {
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'total_samples': len(train_df) + len(test_df),
        'train_label_distribution': train_df['anomaly_type'].value_counts().to_dict(),
        'test_label_distribution': test_df['anomaly_type'].value_counts().to_dict(),
        'feature_count': len(train_df.columns) - 1,  # 减去标签列
        'feature_columns': [col for col in train_df.columns if col != 'anomaly_type']
    }
    
    import json
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"数据统计信息已保存到: {stats_path}")
    
    return train_path, test_path

def main():
    """主函数"""
    logger.info("开始数据集整合流程...")
    
    try:
        # 1. 整合数据集
        merged_df = merge_datasets()
        
        # 2. 分析整体平衡性
        analyze_dataset_balance(merged_df, "合并后数据集")
        
        # 3. 划分训练集和测试集
        train_df, test_df = split_train_test(merged_df, test_ratio=0.2)
        
        # 4. 分析训练集和测试集平衡性
        analyze_dataset_balance(train_df, "训练集")
        analyze_dataset_balance(test_df, "测试集")
        
        # 5. 保存数据集
        train_path, test_path = save_datasets(train_df, test_df)
        
        logger.info("=== 数据集整合完成 ===")
        logger.info(f"训练集: {train_path}")
        logger.info(f"测试集: {test_path}")
        logger.info(f"总样本数: {len(train_df) + len(test_df):,}")
        logger.info(f"训练样本数: {len(train_df):,}")
        logger.info(f"测试样本数: {len(test_df):,}")
        
        # 6. 显示特征信息
        feature_columns = [col for col in train_df.columns if col != 'anomaly_type']
        logger.info(f"特征数量: {len(feature_columns)}")
        logger.info(f"前10个特征: {feature_columns[:10]}")
        
    except Exception as e:
        logger.error(f"数据集整合失败: {e}")
        raise

if __name__ == "__main__":
    main()
