# -*- coding: utf-8 -*-
"""
智能数据集整合脚本
自动处理数据不平衡，对过大文件进行采样，以合适比例合并
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys

print("=" * 70)
print("智能数据集整合脚本 - 自动平衡数据比例")
print("=" * 70)

# 配置
data_dir = r"D:\data1"
output_dir = 'data/processed'

# 目标样本数配置（每类的最大样本数）
MAX_SAMPLES_PER_CLASS = {
    'normal': 80000,        # 正常数据可以多一些
    'brute_force': 15000,   # 暴力破解
    'spoofing': 50000,      # 欺骗攻击
    'upload_attack': 15000, # 上传攻击
    'database_attack': 15000 # 数据库攻击
}

# 最小样本数（如果某类太少，会保留所有）
MIN_SAMPLES_PER_CLASS = 1000

print(f"\n配置:")
print(f"  正常数据最大样本数: {MAX_SAMPLES_PER_CLASS['normal']:,}")
print(f"  攻击类型最大样本数: {MAX_SAMPLES_PER_CLASS['brute_force']:,}")
print(f"  最小样本数: {MIN_SAMPLES_PER_CLASS:,}")

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

# 加载并分析每个文件
file_stats = []
all_dfs = []

for filename, filepath in files:
    print(f"\n3. 加载并分析: {filename}")
    
    try:
        # 读取数据（先读取少量行估算）
        print(f"   读取文件中...")
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
        
        # 记录统计信息
        file_stats.append({
            'filename': filename,
            'attack_type': attack_type,
            'original_count': len(df),
            'target_max': MAX_SAMPLES_PER_CLASS.get(attack_type, 10000),
            'target_min': MIN_SAMPLES_PER_CLASS
        })
        
        # 添加标签
        df['anomaly_type'] = attack_type
        df['Label'] = label
        
        # 智能采样
        target_count = MAX_SAMPLES_PER_CLASS.get(attack_type, 10000)
        current_count = len(df)
        
        if current_count > target_count:
            # 如果数据太多，随机采样
            print(f"   样本数 {current_count:,} > 目标 {target_count:,}，进行下采样")
            df = df.sample(n=target_count, random_state=42)
            print(f"   采样后: {len(df):,} 样本")
        elif current_count < MIN_SAMPLES_PER_CLASS:
            print(f"   警告: 样本数 {current_count:,} < 最小要求 {MIN_SAMPLES_PER_CLASS:,}，保留所有样本")
        else:
            print(f"   样本数 {current_count:,} 在合理范围内，保留所有样本")
        
        all_dfs.append(df)
        
    except Exception as e:
        print(f"   错误: 加载失败 - {e}")
        continue

if not all_dfs:
    print("\n错误: 没有成功加载任何数据!")
    sys.exit(1)

# 按类别合并（如果有多个文件属于同一类别）
print(f"\n4. 按类别合并数据")
category_dfs = {}
for df in all_dfs:
    attack_type = df['anomaly_type'].iloc[0]
    if attack_type not in category_dfs:
        category_dfs[attack_type] = []
    category_dfs[attack_type].append(df)

# 合并同类别数据，并检查是否超过目标
final_dfs = []
for attack_type, dfs in category_dfs.items():
    if len(dfs) > 1:
        merged = pd.concat(dfs, ignore_index=True)
        print(f"   合并 {len(dfs)} 个 {attack_type} 文件")
    else:
        merged = dfs[0]
    
    target_max = MAX_SAMPLES_PER_CLASS.get(attack_type, 10000)
    if len(merged) > target_max:
        print(f"   {attack_type}: {len(merged):,} > 目标 {target_max:,}，再次采样")
        merged = merged.sample(n=target_max, random_state=42)
    
    final_dfs.append(merged)
    print(f"   {attack_type}: {len(merged):,} 样本")

print(f"\n5. 合并所有类别")
merged_df = pd.concat(final_dfs, ignore_index=True)
print(f"   合并后总形状: {merged_df.shape}")

# 显示最终标签分布
print(f"\n6. 最终标签分布:")
label_counts = merged_df['anomaly_type'].value_counts().sort_index()
for label, count in label_counts.items():
    pct = count / len(merged_df) * 100
    print(f"   {label:20s}: {count:8,} ({pct:5.2f}%)")

# 计算平衡性
min_count = label_counts.min()
max_count = label_counts.max()
balance_ratio = min_count / max_count
print(f"\n   平衡比例: {balance_ratio:.3f} (1.0为完全平衡)")

if balance_ratio > 0.5:
    print("   数据集相对平衡")
elif balance_ratio > 0.2:
    print("   数据集轻度不平衡，可以使用类别权重")
else:
    print("   数据集仍不平衡，建议使用类别权重或进一步采样")

# 删除非数值列
print(f"\n7. 清理数据")
non_numeric_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
for col in non_numeric_cols:
    if col in merged_df.columns:
        merged_df = merged_df.drop(columns=[col])
        print(f"   删除列: {col}")

print(f"   清理后形状: {merged_df.shape}")

# 处理无穷大值和缺失值
print(f"\n8. 处理数据质量问题")
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['anomaly_type', 'Label']]

fixed_inf = 0
fixed_nan = 0
for col in numeric_cols:
    # 处理无穷大
    inf_mask = np.isinf(merged_df[col])
    if inf_mask.any():
        finite_values = merged_df[col][~inf_mask]
        if len(finite_values) > 0:
            max_finite = finite_values.max()
            merged_df.loc[inf_mask, col] = max_finite
            fixed_inf += inf_mask.sum()
    
    # 处理缺失值
    nan_count = merged_df[col].isnull().sum()
    if nan_count > 0:
        median_val = merged_df[col].median()
        merged_df[col] = merged_df[col].fillna(median_val)
        fixed_nan += nan_count

print(f"   修复了 {fixed_inf:,} 个无穷大值")
print(f"   修复了 {fixed_nan:,} 个缺失值")

# 划分训练测试集
print(f"\n9. 划分训练/测试集 (80/20, 分层抽样)")
train_df, test_df = train_test_split(
    merged_df,
    test_size=0.2,
    random_state=42,
    stratify=merged_df['anomaly_type']
)

print(f"   训练集: {len(train_df):,} ({len(train_df)/len(merged_df)*100:.1f}%)")
print(f"   测试集: {len(test_df):,} ({len(test_df)/len(merged_df)*100:.1f}%)")

print(f"\n10. 训练集标签分布:")
for label, count in train_df['anomaly_type'].value_counts().sort_index().items():
    pct = count / len(train_df) * 100
    print(f"   {label:20s}: {count:8,} ({pct:5.2f}%)")

print(f"\n11. 测试集标签分布:")
for label, count in test_df['anomaly_type'].value_counts().sort_index().items():
    pct = count / len(test_df) * 100
    print(f"   {label:20s}: {count:8,} ({pct:5.2f}%)")

# 保存
print(f"\n12. 保存数据集")
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, 'train.xlsx')
test_path = os.path.join(output_dir, 'test.xlsx')

print(f"   保存训练集到: {train_path}")
train_df.to_excel(train_path, index=False)

print(f"   保存测试集到: {test_path}")
test_df.to_excel(test_path, index=False)

# 保存统计信息
import json
stats = {
    'total_samples': len(merged_df),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'label_distribution': label_counts.to_dict(),
    'train_distribution': train_df['anomaly_type'].value_counts().to_dict(),
    'test_distribution': test_df['anomaly_type'].value_counts().to_dict(),
    'balance_ratio': balance_ratio,
    'max_samples_config': MAX_SAMPLES_PER_CLASS,
    'file_stats': file_stats
}

stats_path = os.path.join(output_dir, 'merge_statistics.json')
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"   统计信息保存到: {stats_path}")

print("\n" + "=" * 70)
print("✅ 数据集整合完成！")
print("=" * 70)
print(f"总样本数: {len(merged_df):,}")
print(f"平衡比例: {balance_ratio:.3f}")
print(f"训练集: {train_path}")
print(f"测试集: {test_path}")
print(f"\n提示: 如果数据仍不平衡，可以调整MAX_SAMPLES_PER_CLASS配置")
print("=" * 70)
