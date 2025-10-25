#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量检查脚本
检查数据中的无穷大值、极大值和缺失值
"""
import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_quality(df, dataset_name):
    """检查数据质量"""
    logger.info(f"检查数据集: {dataset_name}")
    logger.info(f"数据形状: {df.shape}")
    
    # 获取数值列（排除标签列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'anomaly_type' in numeric_cols:
        numeric_cols.remove('anomaly_type')
    
    logger.info(f"数值列数量: {len(numeric_cols)}")
    
    # 检查无穷大值
    inf_issues = []
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_issues.append((col, inf_count))
            logger.warning(f"列 {col}: {inf_count} 个无穷大值")
    
    # 检查极大值
    large_value_issues = []
    for col in numeric_cols:
        max_val = df[col].max()
        if max_val > 1e10:
            large_value_issues.append((col, max_val))
            logger.warning(f"列 {col}: 最大值 = {max_val}")
    
    # 检查缺失值
    missing_issues = []
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_issues.append((col, missing_count))
            logger.warning(f"列 {col}: {missing_count} 个缺失值")
    
    return {
        'inf_issues': inf_issues,
        'large_value_issues': large_value_issues,
        'missing_issues': missing_issues
    }

def fix_data_issues(df):
    """修复数据问题"""
    logger.info("开始修复数据问题...")
    
    df_fixed = df.copy()
    
    # 获取数值列（排除标签列）
    numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns.tolist()
    if 'anomaly_type' in numeric_cols:
        numeric_cols.remove('anomaly_type')
    
    # 处理无穷大值
    for col in numeric_cols:
        inf_mask = np.isinf(df_fixed[col])
        if inf_mask.any():
            # 用该列的最大有限值替换无穷大值
            finite_values = df_fixed[col][~inf_mask]
            if len(finite_values) > 0:
                max_finite = finite_values.max()
                df_fixed.loc[inf_mask, col] = max_finite
                logger.info(f"列 {col}: 用 {max_finite} 替换无穷大值")
            else:
                # 如果所有值都是无穷大，用0替换
                df_fixed.loc[inf_mask, col] = 0
                logger.info(f"列 {col}: 用 0 替换所有无穷大值")
    
    # 处理极大值（使用对数变换或截断）
    for col in numeric_cols:
        max_val = df_fixed[col].max()
        if max_val > 1e10:
            # 使用99.9%分位数作为截断值
            threshold = df_fixed[col].quantile(0.999)
            df_fixed.loc[df_fixed[col] > threshold, col] = threshold
            logger.info(f"列 {col}: 将大于 {threshold} 的值截断为 {threshold}")
    
    # 处理缺失值
    for col in numeric_cols:
        if df_fixed[col].isnull().any():
            # 用中位数填充
            median_val = df_fixed[col].median()
            df_fixed[col] = df_fixed[col].fillna(median_val)
            logger.info(f"列 {col}: 用中位数 {median_val} 填充缺失值")
    
    return df_fixed

def main():
    """主函数"""
    datasets = {
        'train': 'data/processed/train_processed.xlsx',
        'test': 'data/processed/test_processed.xlsx',
        'train_undersample': 'data/processed/train_undersample_processed.xlsx',
        'test_undersample': 'data/processed/test_undersample_processed.xlsx'
    }
    
    for name, path in datasets.items():
        try:
            logger.info(f"\n=== 处理数据集: {name} ===")
            
            # 读取数据
            df = pd.read_excel(path)
            
            # 检查数据质量
            issues = check_data_quality(df, name)
            
            # 如果有问题，修复数据
            if (issues['inf_issues'] or 
                issues['large_value_issues'] or 
                issues['missing_issues']):
                
                logger.info("发现数据问题，开始修复...")
                df_fixed = fix_data_issues(df)
                
                # 保存修复后的数据
                output_path = path.replace('.xlsx', '_fixed.xlsx')
                df_fixed.to_excel(output_path, index=False)
                logger.info(f"已保存修复后的数据到: {output_path}")
                
                # 再次检查修复后的数据
                logger.info("检查修复后的数据质量...")
                check_data_quality(df_fixed, f"{name}_fixed")
            else:
                logger.info("数据质量良好，无需修复")
                
        except Exception as e:
            logger.error(f"处理数据集 {name} 失败: {e}")

if __name__ == "__main__":
    main()
