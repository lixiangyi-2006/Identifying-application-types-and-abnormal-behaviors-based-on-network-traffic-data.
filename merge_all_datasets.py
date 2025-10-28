#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合数据集脚本
读取 D:\data1 中的所有数据文件，合并并按要求比例划分训练测试集
"""
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_directory(data_dir):
    """分析数据目录中的文件"""
    logger.info(f"扫描数据目录: {data_dir}")
    
    files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(data_dir, filename)
            files.append(filepath)
            logger.info(f"发现文件: {filename}")
    
    return files

def load_and_label_data(filepath):
    """加载数据并添加标签"""
    try:
        logger.info(f"加载文件: {filepath}")
        
        # 读取数据
        df = pd.read_excel(filepath)
        
        # 根据文件名确定标签
        filename = os.path.basename(filepath).lower()
        
        if 'benign' in filename or '良性' in filename or '正常' in filename:
            label = 'normal'
            attack_type = 'normal'
        elif 'brute' in filename or '暴力' in filename or '密码' in filename:
            label = 1
            attack_type = 'brute_force'
        elif 'spoof' in filename or '欺骗' in filename or '伪' in filename:
            label = 1
            attack_type = 'spoofing'
        elif 'upload' in filename or '上传' in filename or '上传危机' in filename:
            label = 1
            attack_type = 'upload_attack'
        elif 'database' in filename or '数据库' in filename:
            label = 1
            attack_type = 'database_attack'
        else:
            logger.warning(f"无法识别文件 {filepath} 的标签，默认标记为 normal")
            label = 0
            attack_type = 'normal'
        
        # 添加标签列
        df['Label'] = label if attack_type != 'normal' else 0
        df['anomaly_type'] = attack_type
        
        logger.info(f"  数据形状: {df.shape}")
        logger.info(f"  标签: {attack_type}")
        logger.info(f"  标签分布: {df['anomaly_type'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"加载文件失败 {filepath}: {e}")
        return None

def balance_classes(df, target_samples_per_class=50000):
    """平衡各类别的样本数量"""
    logger.info(f"开始平衡类别，目标每类样本数: {target_samples_per_class}")
    
    balanced_dfs = []
    
    for attack_type in df['anomaly_type'].unique():
        attack_df = df[df['anomaly_type'] == attack_type].copy()
        
        current_count = len(attack_df)
        logger.info(f"{attack_type}: {current_count} 样本")
        
        if current_count > target_samples_per_class:
            # 如果样本太多，随机采样
            attack_df = attack_df.sample(n=target_samples_per_class, random_state=42)
            logger.info(f"  -> 下采样到 {len(attack_df)} 样本")
        elif current_count < target_samples_per_class:
            # 如果样本太少，随机重复采样
            needed = target_samples_per_class - current_count
            additional_samples = attack_df.sample(n=needed, replace=True, random_state=42)
            attack_df = pd.concat([attack_df, additional_samples], ignore_index=True)
            logger.info(f"  -> 上采样到 {len(attack_df)} 样本")
        else:
            logger.info(f"  -> 保持 {len(attack_df)} 样本")
        
        balanced_dfs.append(attack_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)

def merge_and_split_datasets(data_dir, output_dir='data/processed', 
                            test_size=0.2, balance_classes=False):
    """合并数据集并划分为训练测试集"""
    logger.info("开始整合数据集...")
    
    # 1. 读取所有数据文件
    files = analyze_data_directory(data_dir)
    
    if not files:
        logger.error(f"在 {data_dir} 中没有找到数据文件")
        return
    
    # 2. 加载所有数据
    all_dataframes = []
    
    for filepath in files:
        df = load_and_label_data(filepath)
        if df is not None:
            all_dataframes.append(df)
    
    if not all_dataframes:
        logger.error("没有成功加载任何数据")
        return
    
    # 3. 合并所有数据
    logger.info("合并所有数据集...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 删除非数值列
    non_numeric_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    for col in non_numeric_cols:
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])
    
    logger.info(f"合并后数据形状: {merged_df.shape}")
    
    # 显示数据分布
    logger.info("\n标签分布:")
    label_counts = merged_df['anomaly_type'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(merged_df) * 100
        logger.info(f"  {label}: {count} ({pct:.2f}%)")
    
    # 4. 平衡类别（如果需要）
    if balance_classes:
        logger.info("\n开始平衡类别...")
        merged_df = balance_classes(merged_df)
        logger.info(f"平衡后数据形状: {merged_df.shape}")
    
    # 5. 处理无穷大值
    logger.info("\n处理无穷大值和缺失值...")
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['anomaly_type']:
            # 替换无穷大为该列的最大有限值
            finite_values = merged_df[col][~np.isinf(merged_df[col])]
            if len(finite_values) > 0:
                max_finite = finite_values.max()
                merged_df.loc[np.isinf(merged_df[col]), col] = max_finite
            
            # 处理缺失值
            if merged_df[col].isnull().any():
                median_val = merged_df[col].median()
                merged_df[col] = merged_df[col].fillna(median_val)
    
    # 6. 划分训练测试集
    logger.info("\n划分训练测试集...")
    
    # 使用分层采样确保各类别比例一致
    train_df, test_df = train_test_split(
        merged_df,
        test_size=test_size,
        random_state=42,
        stratify=merged_df['anomaly_type']
    )
    
    logger.info(f"训练集大小: {len(train_df)} ({len(train_df)/len(merged_df)*100:.1f}%)")
    logger.info(f"测试集大小: {len(test_df)} ({len(test_df)/len(merged_df)*100:.1f}%)")
    
    # 显示各数据集中的类别分布
    logger.info("\n训练集标签分布:")
    for label, count in train_df['anomaly_type'].value_counts().items():
        pct = count / len(train_df) * 100
        logger.info(f"  {label}: {count} ({pct:.2f}%)")
    
    logger.info("\n测试集标签分布:")
    for label, count in test_df['anomaly_type'].value_counts().items():
        pct = count / len(test_df) * 100
        logger.info(f"  {label}: {count} ({pct:.2f}%)")
    
    # 7. 保存数据集
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.xlsx')
    test_path = os.path.join(output_dir, 'test.xlsx')
    
    logger.info(f"\n保存训练集到: {train_path}")
    train_df.to_excel(train_path, index=False)
    
    logger.info(f"保存测试集到: {test_path}")
    test_df.to_excel(test_path, index=False)
    
    logger.info("\n数据集整合完成！")
    logger.info(f"训练集: {train_path} (形状: {train_df.shape})")
    logger.info(f"测试集: {test_path} (形状: {test_df.shape})")

def main():
    """主函数"""
    try:
        # 整合数据集
        merge_and_split_datasets(
            data_dir=r"D:\data1",
            output_dir='data/processed',
            test_size=0.2,
            balance_classes=False  # 设置为True可以平衡各类别样本数
        )
        
        logger.info("\n✅ 数据集整合完成！")
        logger.info("现在可以运行 train_balanced_model.py 来训练模型")
        
    except Exception as e:
        logger.error(f"整合数据集失败: {e}")
        raise

if __name__ == "__main__":
    main()
