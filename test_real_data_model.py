#!/usr/bin/env python3
"""
测试真实数据训练的模型性能
验证异常数据分类功能
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.anomaly_detection.inferencer import TwoStageAnomalyDetector
from src.data_adapter import RealDataAdapter
from shared.config import MODEL_DIR, PROCESSED_DATA_DIR
from shared.utils import setup_logging

def test_real_data_model():
    """测试真实数据训练的模型"""
    logger = setup_logging()
    logger.info("=== 测试真实数据训练的模型 ===")
    
    try:
        # 1. 加载真实测试数据
        adapter = RealDataAdapter()
        train_df, test_df = adapter.load_and_preprocess("data/train.xlsx", "data/test.xlsx")
        
        logger.info(f"测试集形状: {test_df.shape}")
        logger.info(f"测试集标签分布:")
        print(test_df['anomaly_type'].value_counts())
        
        # 2. 查找最新训练的模型
        model_files = []
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.startswith("xgboost_real_data_model_") and file.endswith(".pkl"):
                    model_files.append(os.path.join(MODEL_DIR, file))
        
        if not model_files:
            raise FileNotFoundError("未找到真实数据训练的模型文件")
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"使用模型: {latest_model}")
        
        # 3. 创建检测器
        detector = TwoStageAnomalyDetector(latest_model)
        
        # 4. 准备测试数据
        feature_columns = detector.xgboost_model.feature_columns
        if feature_columns is None:
            raise ValueError("模型特征列未定义")
        
        # 确保特征列存在于测试数据中
        missing_features = [col for col in feature_columns if col not in test_df.columns]
        if missing_features:
            logger.warning(f"测试数据中缺少特征: {missing_features}")
            # 使用可用的特征
            available_features = [col for col in feature_columns if col in test_df.columns]
            X_test = test_df[available_features].values
        else:
            X_test = test_df[feature_columns].values
        
        y_test = test_df['anomaly_type'].values
        
        logger.info(f"测试特征数量: {X_test.shape[1]}")
        logger.info(f"测试样本数量: {X_test.shape[0]}")
        
        # 5. 执行预测
        logger.info("开始预测...")
        results = detector.batch_predict(X_test, confidence_threshold=0.5)
        
        # 6. 分析结果
        logger.info("=== 预测结果分析 ===")
        
        # 基本统计
        total_samples = len(test_df)
        predicted_anomalies = sum(1 for label in results['labels'] if label != 'normal')
        high_confidence = sum(1 for conf in results['confidences'] if conf > 0.8)
        needs_review = sum(1 for label in results['labels'] if label == 'needs_review')
        
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"预测异常数: {predicted_anomalies}")
        logger.info(f"高置信度预测: {high_confidence}")
        logger.info(f"需要人工审核: {needs_review}")
        
        # 预测分布
        from collections import Counter
        prediction_counts = Counter(results['labels'])
        logger.info("预测结果分布:")
        for label, count in prediction_counts.items():
            logger.info(f"  {label}: {count}")
        
        # 7. 计算准确率
        true_labels = y_test.tolist()
        correct_predictions = 0
        detailed_results = []
        
        for i, (true_label, pred_label) in enumerate(zip(true_labels, results['labels'])):
            is_correct = true_label == pred_label
            if is_correct:
                correct_predictions += 1
            
            detailed_results.append({
                'sample_id': i,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': results['confidences'][i],
                'is_correct': is_correct,
                'detection_path': results['detection_paths'][i]
            })
        
        accuracy = correct_predictions / total_samples
        logger.info(f"=== 准确率分析 ===")
        logger.info(f"正确预测: {correct_predictions}/{total_samples}")
        logger.info(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 8. 按类别分析准确率
        logger.info("=== 各类别准确率 ===")
        unique_labels = list(set(true_labels))
        for true_label in unique_labels:
            true_indices = [i for i, label in enumerate(true_labels) if label == true_label]
            if true_indices:
                correct_in_class = sum(1 for i in true_indices if results['labels'][i] == true_label)
                class_accuracy = correct_in_class / len(true_indices)
                logger.info(f"{true_label}: {correct_in_class}/{len(true_indices)} = {class_accuracy:.4f}")
        
        # 9. 置信度分析
        logger.info("=== 置信度分析 ===")
        confidences = results['confidences']
        logger.info(f"平均置信度: {np.mean(confidences):.4f}")
        logger.info(f"置信度标准差: {np.std(confidences):.4f}")
        logger.info(f"最低置信度: {np.min(confidences):.4f}")
        logger.info(f"最高置信度: {np.max(confidences):.4f}")
        
        # 10. 错误分析
        logger.info("=== 错误分析 ===")
        errors = [r for r in detailed_results if not r['is_correct']]
        if errors:
            logger.info(f"错误预测数量: {len(errors)}")
            error_types = Counter([f"{e['true_label']}->{e['predicted_label']}" for e in errors])
            logger.info("错误类型分布:")
            for error_type, count in error_types.items():
                logger.info(f"  {error_type}: {count}")
        else:
            logger.info("所有预测都正确！")
        
        # 11. 保存详细结果
        result_df = test_df.copy()
        result_df['predicted_label'] = results['labels']
        result_df['confidence'] = results['confidences']
        result_df['detection_path'] = results['detection_paths']
        result_df['is_correct'] = [r['is_correct'] for r in detailed_results]
        
        output_path = os.path.join(PROCESSED_DATA_DIR, "real_data_test_results.parquet")
        result_df.to_parquet(output_path, index=False)
        logger.info(f"详细测试结果已保存到: {output_path}")
        
        # 12. 生成测试报告
        test_report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_path': latest_model,
            'test_data_source': 'data/test.xlsx',
            'total_samples': total_samples,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'predicted_anomalies': predicted_anomalies,
            'high_confidence_predictions': high_confidence,
            'needs_review': needs_review,
            'average_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'prediction_distribution': dict(prediction_counts),
            'error_count': len(errors),
            'error_types': dict(error_types) if errors else {},
            'true_label_distribution': dict(Counter(true_labels))
        }
        
        report_path = os.path.join(PROCESSED_DATA_DIR, "real_data_test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        logger.info(f"测试报告已保存到: {report_path}")
        
        return test_report
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始测试真实数据训练的模型...")
    
    try:
        test_report = test_real_data_model()
        
        logger.info("=== 测试完成 ===")
        logger.info(f"总体准确率: {test_report['accuracy']:.4f}")
        logger.info(f"预测异常数: {test_report['predicted_anomalies']}")
        logger.info(f"高置信度预测: {test_report['high_confidence_predictions']}")
        
        if test_report['accuracy'] > 0.8:
            logger.info("✅ 异常数据分类功能表现良好")
        elif test_report['accuracy'] > 0.6:
            logger.info("⚠️ 异常数据分类功能需要改进")
        else:
            logger.info("❌ 异常数据分类功能需要重新训练")
            
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        raise

if __name__ == "__main__":
    main()
