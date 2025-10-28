# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

print("开始整合数据集...")

data_dir = r"D:\data1"
output_dir = 'data/processed'

# 读取所有文件
files = []
for filename in os.listdir(data_dir):
    if filename.endswith('.xlsx'):
        files.append(os.path.join(data_dir, filename))
        print(f"发现文件: {filename}")

print(f"\n共找到 {len(files)} 个文件")

# 根据文件名确定标签
all_dfs = []
for filepath in files:
    filename = os.path.basename(filepath).lower()
    print(f"\n处理: {os.path.basename(filepath)}")
    
    df = pd.read_excel(filepath)
    
    # 确定标签
    if 'benign' in filename or '良性' in filename or '正常' in filename:
        attack_type = 'normal'
    elif 'brute' in filename or '暴力' in filename:
        attack_type = 'brute_force'
    elif 'spoof' in filename or '欺骗' in filename or '伪' in filename:
        attack_type = 'spoofing'
    elif 'upload' in filename or '上传' in filename:
        attack_type = 'upload_attack'
    elif 'database' in filename or '数据库' in filename:
        attack_type = 'database_attack'
    else:
        attack_type = 'normal'
    
    df['anomaly_type'] = attack_type
    df['Label'] = 1 if attack_type != 'normal' else 0
    
    print(f"  形状: {df.shape}, 标签: {attack_type}")
    all_dfs.append(df)

# 合并
merged_df = pd.concat(all_dfs, ignore_index=True)

# 删除非数值列
for col in ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']:
    if col in merged_df.columns:
        merged_df = merged_df.drop(columns=[col])

print(f"\n合并后形状: {merged_df.shape}")

# 检查标签分布
print("\n标签分布:")
print(merged_df['anomaly_type'].value_counts())

# 处理无穷大值
print("\n处理无穷大值...")
for col in merged_df.select_dtypes(include=[np.number]).columns:
    if col not in ['anomaly_type']:
        finite = merged_df[col][~np.isinf(merged_df[col])]
        if len(finite) > 0:
            merged_df.loc[np.isinf(merged_df[col]), col] = finite.max()
        if merged_df[col].isnull().any():
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# 划分训练测试集
print("\n划分训练测试集...")
train_df, test_df = train_test_split(
    merged_df, test_size=0.2, random_state=42, stratify=merged_df['anomaly_type']
)

print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

# 保存
os.makedirs(output_dir, exist_ok=True)
train_df.to_excel(os.path.join(output_dir, 'train.xlsx'), index=False)
test_df.to_excel(os.path.join(output_dir, 'test.xlsx'), index=False)

print("\n完成！")
print(f"训练集保存到: {os.path.join(output_dir, 'train.xlsx')}")
print(f"测试集保存到: {os.path.join(output_dir, 'test.xlsx')}")
