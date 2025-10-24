#!/usr/bin/env python3
"""
异常检测系统主程序
支持训练、评估和推理功能
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.anomaly_detection.trainer import AnomalyDetectionTrainer
from src.anomaly_detection.inferencer import TwoStageAnomalyDetector
from src.anomaly_detection.data_loader import NetworkDataLoader
from shared.config import MODEL_DIR, PROCESSED_DATA_DIR
from shared.utils import setup_logging

def train_model(data_path: str, target_column: str = 'anomaly_type', 
               model_save_path: Optional[str] = None) -> Dict[str, Any]:
    """训练异常检测模型"""
    logger = setup_logging()
    logger.info("开始训练异常检测模型...")
    
    try:
        # 创建训练器
        trainer = AnomalyDetectionTrainer()
        
        # 执行完整训练流水线
        results = trainer.full_training_pipeline(
            data_path=data_path,
            target_column=target_column,
            save_model_path=model_save_path
        )
        
        logger.info("模型训练完成!")
        return results
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

def evaluate_model(model_path: str, test_data_path: str, 
                  target_column: str = 'anomaly_type') -> Dict[str, Any]:
    """评估训练好的模型"""
    logger = setup_logging()
    logger.info("开始评估模型...")
    
    try:
        # 创建数据加载器
        data_loader = NetworkDataLoader()
        
        # 加载测试数据
        test_df = data_loader.load_from_file(test_data_path)
        test_df = data_loader.preprocess_network_data(test_df, target_column)
        
        # 创建推理器
        detector = TwoStageAnomalyDetector(model_path)
        
        # 准备测试数据
        X_test, y_test = detector.xgboost_model.prepare_data(test_df, target_column)
        
        # 评估模型
        evaluation_results = detector.xgboost_model.evaluate(X_test, y_test)
        
        logger.info("模型评估完成!")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        raise

def predict_anomalies(model_path: str, data_path: str, 
                     output_path: Optional[str] = None,
                     lightgbm_model_path: Optional[str] = None) -> Dict[str, Any]:
    """使用训练好的模型进行异常检测"""
    logger = setup_logging()
    logger.info("开始异常检测...")
    
    try:
        # 创建数据加载器
        data_loader = NetworkDataLoader()
        
        # 加载数据
        df = data_loader.load_from_file(data_path)
        df = data_loader.preprocess_network_data(df)
        
        # 创建推理器
        detector = TwoStageAnomalyDetector(model_path, lightgbm_model_path)
        
        # 准备特征数据
        feature_columns = detector.xgboost_model.feature_columns
        if feature_columns is None:
            raise ValueError("模型特征列未定义")
        
        X = df[feature_columns].values
        
        # 执行两阶段检测
        results = detector.batch_predict(X)
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['predicted_label'] = results['labels']
        result_df['confidence'] = results['confidences']
        result_df['detection_path'] = results['detection_paths']
        
        # 保存结果
        if output_path:
            data_loader.save_processed_data(result_df, output_path)
            logger.info(f"预测结果已保存到: {output_path}")
        
        logger.info("异常检测完成!")
        return {
            'results': results,
            'result_dataframe': result_df,
            'summary': {
                'total_samples': len(df),
                'predicted_anomalies': sum(1 for label in results['labels'] if label != 'normal'),
                'high_confidence': sum(1 for conf in results['confidences'] if conf > 0.8),
                'needs_review': sum(1 for label in results['labels'] if label == 'needs_review')
            }
        }
        
    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='网络异常检测系统')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       required=True, help='运行模式')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--model', help='模型文件路径')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--target', default='anomaly_type', help='目标列名')
    parser.add_argument('--lightgbm-model', help='LightGBM模型路径')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='置信度阈值')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            # 训练模式
            results = train_model(
                data_path=args.data,
                target_column=args.target,
                model_save_path=args.model
            )
            print("训练完成!")
            print(f"模型保存路径: {results['model_path']}")
            print(f"测试集准确率: {results['test_results']['metrics']['accuracy']:.4f}")
            
        elif args.mode == 'evaluate':
            # 评估模式
            if not args.model:
                raise ValueError("评估模式需要指定模型路径")
            
            results = evaluate_model(
                model_path=args.model,
                test_data_path=args.data,
                target_column=args.target
            )
            print("评估完成!")
            print(f"准确率: {results['metrics']['accuracy']:.4f}")
            print(f"F1分数: {results['metrics']['f1_macro']:.4f}")
            
        elif args.mode == 'predict':
            # 预测模式
            if not args.model:
                raise ValueError("预测模式需要指定模型路径")
            
            results = predict_anomalies(
                model_path=args.model,
                data_path=args.data,
                output_path=args.output,
                lightgbm_model_path=args.lightgbm_model
            )
            print("预测完成!")
            print(f"总样本数: {results['summary']['total_samples']}")
            print(f"检测到异常: {results['summary']['predicted_anomalies']}")
            print(f"高置信度预测: {results['summary']['high_confidence']}")
            print(f"需要人工审核: {results['summary']['needs_review']}")
            
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
