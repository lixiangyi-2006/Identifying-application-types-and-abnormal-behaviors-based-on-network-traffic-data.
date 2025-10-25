#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
细粒度异常分类模型训练脚本
支持：暴力破解、欺骗、上传危机、数据库攻击分类
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

def train_fine_grained_model():
    """训练细粒度异常分类模型"""
    logger.info("开始训练细粒度异常分类模型...")
    
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
    
    # 5. 训练XGBoost模型
    logger.info("开始训练XGBoost模型...")
    
    # XGBoost参数 - 细粒度分类
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_scaled, y_train_encoded)
    
    logger.info("模型训练完成!")
    
    # 6. 验证集评估
    y_val_pred = model.predict(X_test_scaled)
    y_val_pred_proba = model.predict_proba(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test_encoded, y_val_pred)
    precision = precision_score(y_test_encoded, y_val_pred, average='macro')
    recall = recall_score(y_test_encoded, y_val_pred, average='macro')
    f1 = f1_score(y_test_encoded, y_val_pred, average='macro')
    
    logger.info("验证集结果:")
    logger.info(f"  准确率: {accuracy:.4f}")
    logger.info(f"  精确率: {precision:.4f}")
    logger.info(f"  召回率: {recall:.4f}")
    logger.info(f"  F1分数: {f1:.4f}")
    
    # 7. 测试集评估
    logger.info("在测试集上评估模型...")
    
    test_accuracy = accuracy_score(y_test_encoded, y_val_pred)
    test_precision = precision_score(y_test_encoded, y_val_pred, average='macro')
    test_recall = recall_score(y_test_encoded, y_val_pred, average='macro')
    test_f1 = f1_score(y_test_encoded, y_val_pred, average='macro')
    
    logger.info("测试集结果:")
    logger.info(f"  准确率: {test_accuracy:.4f}")
    logger.info(f"  精确率: {test_precision:.4f}")
    logger.info(f"  召回率: {test_recall:.4f}")
    logger.info(f"  F1分数: {test_f1:.4f}")
    
    # 8. 分类报告
    print("\n" + "="*60)
    print("细粒度异常分类结果")
    print("="*60)
    print(f"数据集规模: {X_train.shape[0]:,} 训练样本, {X_test.shape[0]:,} 测试样本")
    print(f"特征数量: {X_train.shape[1]}")
    print(f"攻击类型: {len(label_encoder.classes_)}")
    print(f"标签分布:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y_train_encoded == i)
        print(f"  {class_name}: {count:,} ({count/len(y_train_encoded)*100:.1f}%)")
    
    print(f"\n验证集指标:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    
    print(f"\n测试集指标:")
    print(f"  准确率: {test_accuracy:.4f}")
    print(f"  精确率: {test_precision:.4f}")
    print(f"  召回率: {test_recall:.4f}")
    print(f"  F1分数: {test_f1:.4f}")
    
    print(f"\n分类报告:")
    print(classification_report(y_test_encoded, y_val_pred, 
                              target_names=label_encoder.classes_))
    
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_test_encoded, y_val_pred))
    
    # 9. 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"data/models/fine_grained_anomaly_model_{timestamp}.pkl"
    
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'training_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'test_metrics': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        }
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f"模型已保存到: {model_path}")
    
    print(f"\n模型文件: {model_path}")
    print("="*60)
    
    return model_path

def main():
    """主函数"""
    try:
        model_path = train_fine_grained_model()
        logger.info("细粒度异常分类模型训练完成!")
        logger.info(f"模型保存路径: {model_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()
