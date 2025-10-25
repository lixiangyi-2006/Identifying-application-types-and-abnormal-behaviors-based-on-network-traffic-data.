#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用XGBoost训练真实数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_adapter import RealDataAdapter
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def direct_train():
    """直接训练XGBoost模型"""
    logger.info("开始使用真实数据直接训练XGBoost模型...")
    
    # 1. 加载和预处理数据
    adapter = RealDataAdapter()
    train_df, test_df = adapter.load_and_preprocess("data/train.xlsx", "data/test.xlsx")
    
    logger.info(f"训练集形状: {train_df.shape}")
    logger.info(f"测试集形状: {test_df.shape}")
    
    # 2. 准备特征和标签
    feature_columns = [col for col in train_df.columns if col != 'anomaly_type']
    X_train = train_df[feature_columns].values
    y_train = train_df['anomaly_type'].values
    
    X_test = test_df[feature_columns].values
    y_test = test_df['anomaly_type'].values
    
    logger.info(f"特征数量: {X_train.shape[1]}")
    logger.info(f"训练样本数: {X_train.shape[0]}")
    logger.info(f"测试样本数: {X_test.shape[0]}")
    
    # 3. 标签编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    logger.info(f"标签映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # 4. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. 分割训练和验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
    )
    
    # 6. 创建和训练XGBoost模型
    logger.info("开始训练XGBoost模型...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # 训练模型
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False
    )
    
    logger.info("模型训练完成!")
    
    # 7. 在验证集上评估
    y_val_pred = model.predict(X_val_split)
    y_val_pred_proba = model.predict_proba(X_val_split)
    
    val_accuracy = accuracy_score(y_val_split, y_val_pred)
    val_precision = precision_score(y_val_split, y_val_pred, average='weighted')
    val_recall = recall_score(y_val_split, y_val_pred, average='weighted')
    val_f1 = f1_score(y_val_split, y_val_pred, average='weighted')
    
    logger.info(f"验证集性能:")
    logger.info(f"  准确率: {val_accuracy:.4f}")
    logger.info(f"  精确率: {val_precision:.4f}")
    logger.info(f"  召回率: {val_recall:.4f}")
    logger.info(f"  F1分数: {val_f1:.4f}")
    
    # 8. 在测试集上评估
    logger.info("在测试集上评估模型...")
    y_test_pred = model.predict(X_test_scaled)
    y_test_pred_proba = model.predict_proba(X_test_scaled)
    
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    test_precision = precision_score(y_test_encoded, y_test_pred, average='weighted')
    test_recall = recall_score(y_test_encoded, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted')
    
    logger.info(f"测试集性能:")
    logger.info(f"  准确率: {test_accuracy:.4f}")
    logger.info(f"  精确率: {test_precision:.4f}")
    logger.info(f"  召回率: {test_recall:.4f}")
    logger.info(f"  F1分数: {test_f1:.4f}")
    
    # 9. 保存模型和预处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"data/models/xgboost_real_data_model_{timestamp}.pkl"
    
    # 保存模型、标签编码器和特征缩放器
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'training_metrics': {
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        },
        'test_metrics': {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f"模型已保存到: {model_path}")
    
    # 10. 生成详细报告
    print("\n" + "="*60)
    print("真实数据训练报告")
    print("="*60)
    print(f"数据集规模: {train_df.shape[0]:,} 训练样本, {test_df.shape[0]:,} 测试样本")
    print(f"特征数量: {X_train.shape[1]}")
    print(f"标签分布:")
    print(f"  正常: {np.sum(y_train == 'normal'):,} ({np.sum(y_train == 'normal')/len(y_train)*100:.1f}%)")
    print(f"  异常: {np.sum(y_train == 'ddos'):,} ({np.sum(y_train == 'ddos')/len(y_train)*100:.1f}%)")
    
    print(f"\n验证集性能:")
    print(f"  准确率: {val_accuracy:.4f}")
    print(f"  精确率: {val_precision:.4f}")
    print(f"  召回率: {val_recall:.4f}")
    print(f"  F1分数: {val_f1:.4f}")
    
    print(f"\n测试集性能:")
    print(f"  准确率: {test_accuracy:.4f}")
    print(f"  精确率: {test_precision:.4f}")
    print(f"  召回率: {test_recall:.4f}")
    print(f"  F1分数: {test_f1:.4f}")
    
    print(f"\n分类报告:")
    print(classification_report(y_test_encoded, y_test_pred, target_names=label_encoder.classes_))
    
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_test_encoded, y_test_pred))
    
    print(f"\n模型文件: {model_path}")
    print("="*60)
    
    return model_path

if __name__ == "__main__":
    try:
        model_path = direct_train()
        print(f"\n训练成功完成! 模型保存在: {model_path}")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
