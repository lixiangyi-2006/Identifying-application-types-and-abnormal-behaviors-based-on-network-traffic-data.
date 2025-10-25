#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
处理非数值特征，为模型训练做准备
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_features(df):
    """预处理特征，处理非数值列"""
    logger.info("开始预处理特征...")
    
    # 复制数据
    df_processed = df.copy()
    
    # 需要删除的非数值列
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']
    
    # 删除非数值列
    for col in columns_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])
            logger.info(f"删除列: {col}")
    
    # 检查是否还有非数值列（除了标签列）
    non_numeric_cols = df_processed.select_dtypes(exclude=['number']).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col != 'anomaly_type']
    
    if non_numeric_cols:
        logger.warning(f"仍有非数值列: {non_numeric_cols}")
        # 尝试转换为数值
        for col in non_numeric_cols:
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                logger.info(f"转换列 {col} 为数值")
            except Exception as e:
                logger.error(f"转换列 {col} 失败: {e}")
                # 如果转换失败，删除该列
                df_processed = df_processed.drop(columns=[col])
                logger.info(f"删除无法转换的列: {col}")
    
    # 处理缺失值
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != 'anomaly_type']
    
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            # 用中位数填充缺失值
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
            logger.info(f"用中位数 {median_val:.2f} 填充列 {col} 的缺失值")
    
    logger.info(f"预处理完成: {df.shape} -> {df_processed.shape}")
    logger.info(f"剩余特征数: {len(df_processed.columns) - 1}")  # 减去标签列
    
    return df_processed

def preprocess_datasets():
    """预处理所有数据集"""
    logger.info("开始预处理数据集...")
    
    datasets = {
        'train': 'data/processed/train.xlsx',
        'test': 'data/processed/test.xlsx',
        'train_undersample': 'data/processed/train_undersample.xlsx',
        'test_undersample': 'data/processed/test_undersample.xlsx'
    }
    
    processed_datasets = {}
    
    for name, path in datasets.items():
        if pd.io.common.file_exists(path):
            logger.info(f"处理数据集: {name}")
            
            # 读取数据
            df = pd.read_excel(path)
            logger.info(f"原始形状: {df.shape}")
            
            # 预处理
            df_processed = preprocess_features(df)
            
            # 保存处理后的数据
            output_path = path.replace('.xlsx', '_processed.xlsx')
            df_processed.to_excel(output_path, index=False)
            logger.info(f"已保存到: {output_path}")
            
            processed_datasets[name] = {
                'path': output_path,
                'shape': df_processed.shape,
                'features': len(df_processed.columns) - 1
            }
        else:
            logger.warning(f"文件不存在: {path}")
    
    return processed_datasets

def main():
    """主函数"""
    try:
        processed_datasets = preprocess_datasets()
        
        logger.info("\n=== 数据预处理完成 ===")
        logger.info("处理后的数据集:")
        for name, info in processed_datasets.items():
            logger.info(f"  {name}: {info['shape']} (特征数: {info['features']})")
        
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        raise

if __name__ == "__main__":
    main()
