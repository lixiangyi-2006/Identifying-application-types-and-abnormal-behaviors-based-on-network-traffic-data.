# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys

print("=" * 60)
print("数据集整合脚本")
print("=" * 60)

# 数据目录
data_dir = r"D:\data1"
output_dir = 'data/processed'

print(f"\n1. 扫描数据目录: {data_dir}")

# 获取所有Excel文件
files = []
for filename in os.listdir(data_dir):
    if filename.endswith('.xlsx'):
        files.append((filename, os.path.join(data_dir, filename)))
        print(f"   发现: {filename}")

if not files:
    print("错误: 没有找到Excel文件!")
    sys.exit(1)

print(f"\n2. 共找到 {len(files)} 个文件")

# 加载并标记数据
all_dfs = []
for filename, filepath in files:
    print(f"\n3. 加载: {filename}")
    
    try:
        df = pd.read_excel(filepath)
        print(f"   原始形状: {df.shape}")
        
        # 确定标签
        filename_lower = filename.lower()
        if '良性' in filename or '正常' in filename:
            attack_type = 'normal'
            label = 0
        elif '暴力' in filename or '密码' in filename:
            attack_type = 'brute_force'
            label = 1
        elif '欺骗' in filename or '伪' in filename:
            attack_type = 'spoofing'
            label = 1
        elif '上传' in filename:
            attack_type = 'upload_attack'
            label = 1
        elif '数据库' in filename:
            attack_type = 'database_attack'
            label = 1
        else:
            print(f"   警告: 无法识别文件类型，默认标记为normal")
            attack_type = 'normal'
            label = 0
        
        # 添加标签
        df['anomaly_type'] = attack_type
        df['Label'] = label
        
        print(f"   标记为: {attack_type}")
        print(f"   当前形状: {df.shape}")
        
        all_dfs.append(df)
        
    except Exception as e:
        print(f"   错误: 加载失败 - {e}")
        continue

if not all_dfs:
    print("\n错误: 没有成功加载任何数据!")
    sys.exit(1)

print(f"\n4. 合并所有数据集")
merged_df = pd.concat(all_dfs, ignore_index=True)
print(f"   合并后形状: {merged_df.shape}")

# 删除非数值列
print(f"\n5. 清理数据")
non_numeric_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
for col in non_numeric_cols:
    if col in merged_df.columns:
        merged_df = merged_df.drop(columns=[col])
        print(f"   删除列: {col}")

print(f"   清理后形状: {merged_df.shape}")

# 显示标签分布
print(f"\n6. 标签分布:")
label_counts = merged_df['anomaly_type'].value_counts()
for label, count in label_counts.items():
    pct = count / len(merged_df) * 100
    print(f"   {label}: {count:,} ({pct:.2f}%)")

# 处理无穷大值
print(f"\n7. 处理数据质量问题")
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['anomaly_type', 'Label']]

fixed_count = 0
for col in numeric_cols:
    # 处理无穷大
    inf_mask = np.isinf(merged_df[col])
    if inf_mask.any():
        finite_values = merged_df[col][~inf_mask]
        if len(finite_values) > 0:
            max_finite = finite_values.max()
            merged_df.loc[inf_mask, col] = max_finite
            fixed_count += inf_mask.sum()
    
    # 处理缺失值
    if merged_df[col].isnull().any():
        median_val = merged_df[col].median()
        merged_df[col] = merged_df[col].fillna(median_val)

print(f"   修复了 {fixed_count} 个无穷大值")

# 划分训练测试集
print(f"\n8. 划分训练/测试集 (80/20)")
train_df, test_df = train_test_split(
    merged_df,
    test_size=0.2,
    random_state=42,
    stratify=merged_df['anomaly_type']
)

print(f"   训练集: {len(train_df):,} ({len(train_df)/len(merged_df)*100:.1f}%)")
print(f"   测试集: {len(test_df):,} ({len(test_df)/len(merged_df)*100:.1f}%)")

print(f"\n9. 训练集标签分布:")
for label, count in train_df['anomaly_type'].value_counts().items():
    pct = count / len(train_df) * 100
    print(f"   {label}: {count:,} ({pct:.2f}%)")

print(f"\n10. 测试集标签分布:")
for label, count in test_df['anomaly_type'].value_counts().items():
    pct = count / len(test_df) * 100
    print(f"   {label}: {count:,} ({pct:.2f}%)")

# 保存
print(f"\n11. 保存数据集")
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, 'train.xlsx')
test_path = os.path.join(output_dir, 'test.xlsx')

print(f"   保存训练集到: {train_path}")
train_df.to_excel(train_path, index=False)

print(f"   保存测试集到: {test_path}")
test_df.to_excel(test_path, index=False)

print("\n" + "=" * 60)
print("✅ 数据集整合完成！")
print("=" * 60)
print(f"总样本数: {len(merged_df):,}")
print(f"训练集: {train_path}")
print(f"测试集: {test_path}")
print("\n可以运行 train_balanced_model.py 来训练模型")
print("=" * 60)
