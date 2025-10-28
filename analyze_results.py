#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的细粒度异常分类结果分析
基于训练脚本的输出结果进行分析
"""
import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_training_results():
    """分析训练结果"""
    logger.info("分析细粒度异常分类训练结果...")
    
    # 基于训练脚本的输出结果
    results = {
        'original': {
            'accuracy': 0.7715,
            'precision': 0.5107,
            'recall': 0.7377,
            'f1_score': 0.5139,
            'class_performance': {
                'brute_force': {'precision': 0.09, 'recall': 0.83, 'f1': 0.17},
                'database_attack': {'precision': 0.48, 'recall': 0.73, 'f1': 0.58},
                'normal': {'precision': 0.97, 'recall': 0.84, 'f1': 0.90},
                'spoofing': {'precision': 0.89, 'recall': 0.62, 'f1': 0.73},
                'upload_attack': {'precision': 0.11, 'recall': 0.68, 'f1': 0.19}
            }
        },
        'undersample': {
            'accuracy': 0.4689,
            'precision': 0.4364,
            'recall': 0.6447,
            'f1_score': 0.3152,
            'class_performance': {
                'brute_force': {'precision': 0.05, 'recall': 0.89, 'f1': 0.10},
                'database_attack': {'precision': 0.12, 'recall': 0.74, 'f1': 0.21},
                'normal': {'precision': 1.00, 'recall': 0.49, 'f1': 0.66},
                'spoofing': {'precision': 0.96, 'recall': 0.35, 'f1': 0.52},
                'upload_attack': {'precision': 0.05, 'recall': 0.74, 'f1': 0.09}
            }
        }
    }
    
    print("=" * 80)
    print("细粒度异常分类模型训练结果分析")
    print("=" * 80)
    
    # 比较两种策略
    print("\n策略对比:")
    print("-" * 60)
    print(f"{'指标':<15} {'Original策略':<15} {'Undersample策略':<15} {'推荐':<10}")
    print("-" * 60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    for metric, name in zip(metrics, metric_names):
        orig_val = results['original'][metric]
        under_val = results['undersample'][metric]
        better = 'Original' if orig_val > under_val else 'Undersample'
        print(f"{name:<15} {orig_val:<15.4f} {under_val:<15.4f} {better:<10}")
    
    print("-" * 60)
    print(f"推荐策略: Original (F1分数更高: {results['original']['f1_score']:.4f})")
    
    # 详细分析最佳模型（Original策略）
    print("\n" + "=" * 60)
    print("最佳模型详细分析 (Original策略)")
    print("=" * 60)
    
    best_model = results['original']
    
    print(f"\n整体性能:")
    print(f"  准确率: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.1f}%)")
    print(f"  精确率: {best_model['precision']:.4f}")
    print(f"  召回率: {best_model['recall']:.4f}")
    print(f"  F1分数: {best_model['f1_score']:.4f}")
    
    # 各类别性能分析
    print(f"\n各类别性能分析:")
    print("-" * 50)
    
    attack_types = {
        'brute_force': '暴力破解',
        'database_attack': '数据库攻击',
        'spoofing': '欺骗攻击',
        'upload_attack': '上传攻击',
        'normal': '正常流量'
    }
    
    for attack_type, chinese_name in attack_types.items():
        perf = best_model['class_performance'][attack_type]
        print(f"\n{chinese_name} ({attack_type}):")
        print(f"  精确率: {perf['precision']:.4f} ({perf['precision']*100:.1f}%)")
        print(f"  召回率: {perf['recall']:.4f} ({perf['recall']*100:.1f}%)")
        print(f"  F1分数: {perf['f1']:.4f}")
        
        # 性能评估
        if perf['f1'] >= 0.7:
            status = "[优秀]"
        elif perf['f1'] >= 0.5:
            status = "[良好]"
        elif perf['f1'] >= 0.3:
            status = "[一般]"
        else:
            status = "[需要改进]"
        
        print(f"  评估: {status}")
    
    # 攻击检测能力总结
    print(f"\n" + "=" * 50)
    print("攻击检测能力总结")
    print("=" * 50)
    
    attack_detection = {
        '暴力破解': best_model['class_performance']['brute_force']['f1'],
        '数据库攻击': best_model['class_performance']['database_attack']['f1'],
        '欺骗攻击': best_model['class_performance']['spoofing']['f1'],
        '上传攻击': best_model['class_performance']['upload_attack']['f1']
    }
    
    print("各类攻击检测F1分数:")
    for attack_name, f1_score in attack_detection.items():
        print(f"  {attack_name}: {f1_score:.4f}")
    
    # 正常流量识别
    normal_f1 = best_model['class_performance']['normal']['f1']
    print(f"\n正常流量识别F1分数: {normal_f1:.4f}")
    
    # 总体评估
    print(f"\n" + "=" * 50)
    print("总体评估")
    print("=" * 50)
    
    # 检查是否满足细粒度分类要求
    requirements_met = True
    poor_performance = []
    
    for attack_name, f1_score in attack_detection.items():
        if f1_score < 0.3:
            requirements_met = False
            poor_performance.append(attack_name)
    
    if requirements_met:
        print("[成功] 模型成功满足细粒度异常分类要求！")
        print("   能够有效区分不同类型的网络攻击")
        print("   整体准确率达到77.15%，F1分数为51.39%")
    else:
        print("[部分满足] 模型部分满足要求，但以下攻击类型检测效果需要改进:")
        for attack in poor_performance:
            print(f"   - {attack}")
    
    # 模型优势
    print(f"\n模型优势:")
    print(f"  [优势] 正常流量识别准确率高 (F1: {normal_f1:.4f})")
    print(f"  [优势] 欺骗攻击检测效果好 (F1: {attack_detection['欺骗攻击']:.4f})")
    print(f"  [优势] 数据库攻击检测良好 (F1: {attack_detection['数据库攻击']:.4f})")
    
    # 需要改进的地方
    print(f"\n需要改进的地方:")
    print(f"  [注意] 暴力破解检测精确率低 (9%)，但召回率高 (83%)")
    print(f"  [注意] 上传攻击检测精确率低 (11%)，但召回率高 (68%)")
    print(f"  [建议] 增加这两类攻击的训练样本，优化特征工程")
    
    # 实际应用建议
    print(f"\n" + "=" * 50)
    print("实际应用建议")
    print("=" * 50)
    
    print("1. 模型部署:")
    print("   - 使用Original策略训练的模型")
    print("   - 模型文件: data/models/fine_grained_model_original_20251025_171745.pkl")
    
    print("\n2. 检测策略:")
    print("   - 正常流量: 高置信度识别，减少误报")
    print("   - 欺骗攻击: 高精确率，适合实时检测")
    print("   - 数据库攻击: 平衡的检测性能")
    print("   - 暴力破解/上传攻击: 高召回率，需要人工复核")
    
    print("\n3. 监控重点:")
    print("   - 重点关注暴力破解和上传攻击的误报情况")
    print("   - 定期评估模型性能，收集新的攻击样本")
    print("   - 考虑结合规则引擎进行二次验证")
    
    return results

def main():
    """主函数"""
    try:
        results = analyze_training_results()
        logger.info("\n分析完成！")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise

if __name__ == "__main__":
    main()
