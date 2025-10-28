#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
细粒度异常分类测试脚本
测试训练好的模型对五种攻击类型的分类效果
"""
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """加载训练好的模型"""
    logger.info(f"加载模型: {model_path}")
    
    # 直接读取模型文件内容
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # 检查模型数据结构
    logger.info(f"模型数据键: {list(model_data.keys())}")
    
    return model_data

def test_fine_grained_classification():
    """测试细粒度分类效果"""
    logger.info("开始测试细粒度异常分类...")
    
    # 加载最佳模型（original策略）
    model_path = 'data/models/fine_grained_model_original_20251025_171745.pkl'
    model_data = load_model(model_path)
    
    # 加载测试数据
    test_df = pd.read_excel('data/processed/test_processed_fixed.xlsx')
    logger.info(f"测试集形状: {test_df.shape}")
    
    # 准备特征和标签
    feature_cols = [col for col in test_df.columns if col != 'anomaly_type']
    X_test = test_df[feature_cols]
    y_test = test_df['anomaly_type']
    
    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"测试样本数: {len(X_test)}")
    
    # 获取标签映射
    label_mapping = model_data['label_mapping']
    logger.info(f"标签映射: {label_mapping}")
    
    # 预测
    logger.info("开始预测...")
    y_pred = model_data['model'].predict(X_test)
    
    # 计算准确率
    accuracy = (y_pred == y_test).mean()
    logger.info(f"测试准确率: {accuracy:.4f}")
    
    # 生成详细报告
    logger.info("\n=== 细粒度异常分类测试结果 ===")
    
    # 标签名称映射
    label_names = {
        0: 'brute_force (暴力破解)',
        1: 'database_attack (数据库攻击)', 
        2: 'normal (正常)',
        3: 'spoofing (欺骗)',
        4: 'upload_attack (上传攻击)'
    }
    
    # 分类报告
    report = classification_report(y_test, y_pred, 
                                 target_names=[label_names[i] for i in range(5)],
                                 output_dict=True)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, 
                               target_names=[label_names[i] for i in range(5)]))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n混淆矩阵:")
    print("实际\\预测", end="")
    for i in range(5):
        print(f"\t{label_names[i][:10]}", end="")
    print()
    
    for i in range(5):
        print(f"{label_names[i][:10]}", end="")
        for j in range(5):
            print(f"\t{cm[i,j]:6d}", end="")
        print()
    
    # 各类别性能分析
    print("\n=== 各类别性能分析 ===")
    for i, (label, name) in enumerate(label_names.items()):
        precision = report[str(label)]['precision']
        recall = report[str(label)]['recall']
        f1 = report[str(label)]['f1-score']
        support = report[str(label)]['support']
        
        print(f"{name}:")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  样本数: {support}")
        print()
    
    # 攻击类型检测能力分析
    print("=== 攻击类型检测能力分析 ===")
    attack_types = ['brute_force', 'database_attack', 'spoofing', 'upload_attack']
    
    for attack in attack_types:
        label_idx = label_mapping[attack]
        precision = report[str(label_idx)]['precision']
        recall = report[str(label_idx)]['recall']
        f1 = report[str(label_idx)]['f1-score']
        
        print(f"{attack}:")
        print(f"  检测精确率: {precision:.4f} (预测为{attack}的样本中，{precision*100:.1f}%确实是{attack})")
        print(f"  检测召回率: {recall:.4f} (所有{attack}样本中，{recall*100:.1f}%被正确识别)")
        print(f"  综合F1分数: {f1:.4f}")
        print()
    
    # 正常流量识别能力
    normal_idx = label_mapping['normal']
    normal_precision = report[str(normal_idx)]['precision']
    normal_recall = report[str(normal_idx)]['recall']
    normal_f1 = report[str(normal_idx)]['f1-score']
    
    print("=== 正常流量识别能力 ===")
    print(f"正常流量识别精确率: {normal_precision:.4f}")
    print(f"正常流量识别召回率: {normal_recall:.4f}")
    print(f"正常流量F1分数: {normal_f1:.4f}")
    print()
    
    # 总体评估
    print("=== 总体评估 ===")
    print(f"整体准确率: {accuracy:.4f}")
    print(f"宏平均F1分数: {report['macro avg']['f1-score']:.4f}")
    print(f"加权平均F1分数: {report['weighted avg']['f1-score']:.4f}")
    
    # 模型是否满足要求
    print("\n=== 模型是否满足细粒度分类要求 ===")
    requirements_met = True
    
    # 检查各类攻击的F1分数是否达到合理水平（>0.3）
    for attack in attack_types:
        label_idx = label_mapping[attack]
        f1 = report[str(label_idx)]['f1-score']
        if f1 < 0.3:
            print(f"❌ {attack} 的F1分数 {f1:.4f} 低于0.3，需要改进")
            requirements_met = False
        else:
            print(f"✅ {attack} 的F1分数 {f1:.4f} 达到要求")
    
    if requirements_met:
        print("\n🎉 模型成功满足细粒度异常分类要求！")
        print("   能够有效区分暴力破解、欺骗、上传攻击、数据库攻击等不同类型")
    else:
        print("\n⚠️  模型部分满足要求，但某些攻击类型检测效果需要改进")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'requirements_met': requirements_met
    }

def main():
    """主函数"""
    try:
        results = test_fine_grained_classification()
        
        logger.info("\n测试完成！")
        logger.info(f"模型准确率: {results['accuracy']:.4f}")
        logger.info(f"是否满足要求: {results['requirements_met']}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

if __name__ == "__main__":
    main()
