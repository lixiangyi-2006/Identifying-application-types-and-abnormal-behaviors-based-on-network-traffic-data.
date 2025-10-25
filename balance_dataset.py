#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据不平衡处理脚本
提供多种方法解决数据不平衡问题：
1. SMOTE过采样
2. 类别权重平衡
3. 下采样
4. 混合策略
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import logging
from datetime import datetime
import joblib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_imbalance(df, dataset_name="数据集"):
    """分析数据不平衡情况"""
    logger.info(f"=== {dataset_name} 不平衡分析 ===")
    
    label_counts = df['anomaly_type'].value_counts()
    total = len(df)
    
    print(f"总样本数: {total:,}")
    print("各类别分布:")
    for label, count in label_counts.items():
        percentage = count / total * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")
    
    # 计算不平衡指标
    min_count = label_counts.min()
    max_count = label_counts.max()
    imbalance_ratio = min_count / max_count
    ratio_max_min = max_count / min_count
    
    print(f"\n不平衡比例: {imbalance_ratio:.3f}")
    print(f"最大类别/最小类别: {ratio_max_min:.1f}:1")
    
    if imbalance_ratio > 0.5:
        print("数据相对平衡")
    elif imbalance_ratio > 0.2:
        print("数据轻度不平衡")
    elif imbalance_ratio > 0.1:
        print("数据中度不平衡")
    else:
        print("数据严重不平衡")
    
    return {
        'total_samples': total,
        'label_counts': label_counts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'ratio_max_min': ratio_max_min
    }

def apply_smote_oversampling(df, target_column='anomaly_type', random_state=42):
    """使用SMOTE进行过采样"""
    logger.info("开始SMOTE过采样...")
    
    # 分离特征和标签
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[target_column]
    
    # 标准化特征（SMOTE需要数值特征）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 应用SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # 转换回DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
    resampled_df[target_column] = y_resampled
    
    logger.info(f"SMOTE过采样完成: {len(df)} -> {len(resampled_df)}")
    
    return resampled_df, scaler

def apply_undersampling(df, target_column='anomaly_type', random_state=42):
    """使用下采样平衡数据"""
    logger.info("开始下采样...")
    
    # 分离特征和标签
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[target_column]
    
    # 应用下采样
    undersampler = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    
    # 转换回DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
    resampled_df[target_column] = y_resampled
    
    logger.info(f"下采样完成: {len(df)} -> {len(resampled_df)}")
    
    return resampled_df

def apply_smotetomek(df, target_column='anomaly_type', random_state=42):
    """使用SMOTETomek混合采样"""
    logger.info("开始SMOTETomek混合采样...")
    
    # 分离特征和标签
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[target_column]
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 应用SMOTETomek
    smotetomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smotetomek.fit_resample(X_scaled, y)
    
    # 转换回DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
    resampled_df[target_column] = y_resampled
    
    logger.info(f"SMOTETomek混合采样完成: {len(df)} -> {len(resampled_df)}")
    
    return resampled_df, scaler

def create_balanced_dataset_strategy(df, strategy='smote', target_column='anomaly_type'):
    """创建平衡数据集的策略"""
    logger.info(f"使用策略: {strategy}")
    
    if strategy == 'smote':
        return apply_smote_oversampling(df, target_column)
    elif strategy == 'undersample':
        return apply_undersampling(df, target_column)
    elif strategy == 'smotetomek':
        return apply_smotetomek(df, target_column)
    else:
        raise ValueError(f"未知策略: {strategy}")

def calculate_class_weights(df, target_column='anomaly_type'):
    """计算类别权重"""
    logger.info("计算类别权重...")
    
    label_counts = df[target_column].value_counts()
    total_samples = len(df)
    n_classes = len(label_counts)
    
    # 计算平衡权重
    class_weights = {}
    for label, count in label_counts.items():
        weight = total_samples / (n_classes * count)
        class_weights[label] = weight
    
    logger.info("类别权重:")
    for label, weight in class_weights.items():
        print(f"  {label}: {weight:.3f}")
    
    return class_weights

def save_balanced_datasets(train_df, test_df, strategy, output_dir="data/processed"):
    """保存平衡后的数据集"""
    logger.info(f"保存{strategy}策略的数据集...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_path = os.path.join(output_dir, f"train_{strategy}.xlsx")
    train_df.to_excel(train_path, index=False)
    logger.info(f"训练集已保存到: {train_path}")
    
    # 保存测试集
    test_path = os.path.join(output_dir, f"test_{strategy}.xlsx")
    test_df.to_excel(test_path, index=False)
    logger.info(f"测试集已保存到: {test_path}")
    
    # 保存统计信息
    stats = {
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_label_distribution': train_df['anomaly_type'].value_counts().to_dict(),
        'test_label_distribution': test_df['anomaly_type'].value_counts().to_dict()
    }
    
    import json
    stats_path = os.path.join(output_dir, f"dataset_stats_{strategy}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"统计信息已保存到: {stats_path}")
    
    return train_path, test_path

def main():
    """主函数"""
    logger.info("开始数据不平衡处理...")
    
    try:
        # 1. 加载原始数据
        logger.info("加载原始数据集...")
        train_df = pd.read_excel('data/processed/train.xlsx')
        test_df = pd.read_excel('data/processed/test.xlsx')
        
        # 2. 分析原始数据不平衡情况
        logger.info("分析原始数据不平衡情况...")
        original_stats = analyze_imbalance(train_df, "原始训练集")
        
        # 3. 计算类别权重（用于模型训练）
        class_weights = calculate_class_weights(train_df)
        
        # 4. 保存类别权重
        weights_path = "data/processed/class_weights.json"
        import json
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(class_weights, f, indent=2, ensure_ascii=False)
        logger.info(f"类别权重已保存到: {weights_path}")
        
        # 5. 尝试不同的平衡策略
        strategies = ['smote', 'undersample', 'smotetomek']
        
        for strategy in strategies:
            logger.info(f"\n=== 尝试策略: {strategy} ===")
            
            try:
                # 应用平衡策略
                if strategy == 'smote':
                    balanced_train_df, scaler = apply_smote_oversampling(train_df)
                elif strategy == 'undersample':
                    balanced_train_df = apply_undersampling(train_df)
                elif strategy == 'smotetomek':
                    balanced_train_df, scaler = apply_smotetomek(train_df)
                
                # 分析平衡后的数据
                balanced_stats = analyze_imbalance(balanced_train_df, f"{strategy}平衡后训练集")
                
                # 保存平衡后的数据集
                save_balanced_datasets(balanced_train_df, test_df, strategy)
                
                logger.info(f"{strategy}策略处理完成!")
                
            except Exception as e:
                logger.error(f"{strategy}策略处理失败: {e}")
                continue
        
        logger.info("\n=== 数据不平衡处理完成 ===")
        logger.info("生成的文件:")
        logger.info("1. class_weights.json - 类别权重")
        logger.info("2. train_smote.xlsx - SMOTE过采样训练集")
        logger.info("3. train_undersample.xlsx - 下采样训练集")
        logger.info("4. train_smotetomek.xlsx - SMOTETomek混合采样训练集")
        
    except Exception as e:
        logger.error(f"数据不平衡处理失败: {e}")
        raise

if __name__ == "__main__":
    main()
