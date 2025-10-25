#!/usr/bin/env python3
"""
细粒度异常分类模型测试脚本
测试模型对暴力破解、欺骗、上传危机、数据库攻击的分类能力
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_adapter import RealDataAdapter
from shared.config import MODEL_DIR, PROCESSED_DATA_DIR
from shared.utils import setup_logging

def test_fine_grained_model():
    """测试细粒度异常分类模型"""
    logger = setup_logging()
    logger.info("=== 测试细粒度异常分类模型 ===")
    
    try:
        # 1. 加载测试数据
        adapter = RealDataAdapter()
        train_df, test_df = adapter.load_and_preprocess("data/train.xlsx", "data/test.xlsx")
        
        logger.info(f"测试集形状: {test_df.shape}")
        logger.info(f"测试集标签分布:")
        print(test_df['anomaly_type'].value_counts())
        
        # 2. 查找最新的细粒度模型
        model_files = []
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.startswith("fine_grained_anomaly_model_") and file.endswith(".pkl"):
                    model_files.append(os.path.join(MODEL_DIR, file))
        
        if not model_files:
            raise FileNotFoundError("未找到细粒度异常分类模型文件")
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"使用模型: {latest_model}")
        
        # 3. 加载模型
        model_data = joblib.load(latest_model)
        
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        logger.info(f"模型攻击类型: {list(label_encoder.classes_)}")
        logger.info(f"模型特征数量: {len(feature_columns)}")
        
        # 4. 准备测试数据
        X_test = test_df[feature_columns].values
        y_test = test_df['anomaly_type'].values
        
        # 标准化特征
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"测试样本数量: {X_test.shape[0]}")
        
        # 5. 执行预测
        logger.info("开始预测...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # 6. 分析结果
        logger.info("=== 预测结果分析 ===")
        
        # 基本统计
        total_samples = len(test_df)
        max_probs = np.max(y_pred_proba, axis=1)
        high_confidence = sum(1 for conf in max_probs if conf > 0.8)
        
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"高置信度预测: {high_confidence}")
        
        # 预测分布
        from collections import Counter
        prediction_counts = Counter(y_pred)
        logger.info("预测结果分布:")
        for label_id, count in prediction_counts.items():
            label_name = label_encoder.classes_[label_id]
            logger.info(f"  {label_name}: {count}")
        
        # 7. 计算准确率
        correct_predictions = sum(1 for true, pred in zip(y_test, y_pred) if true == label_encoder.classes_[pred])
        accuracy = correct_predictions / total_samples
        
        logger.info(f"=== 准确率分析 ===")
        logger.info(f"正确预测: {correct_predictions}/{total_samples}")
        logger.info(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 8. 按攻击类型分析准确率
        logger.info("=== 各类别准确率 ===")
        unique_labels = list(set(y_test))
        for true_label in unique_labels:
            true_indices = [i for i, label in enumerate(y_test) if label == true_label]
            if true_indices:
                correct_in_class = sum(1 for i in true_indices if y_test[i] == label_encoder.classes_[y_pred[i]])
                class_accuracy = correct_in_class / len(true_indices)
                logger.info(f"{true_label}: {correct_in_class}/{len(true_indices)} = {class_accuracy:.4f}")
        
        # 9. 置信度分析
        logger.info("=== 置信度分析 ===")
        logger.info(f"平均置信度: {np.mean(max_probs):.4f}")
        logger.info(f"置信度标准差: {np.std(max_probs):.4f}")
        logger.info(f"最低置信度: {np.min(max_probs):.4f}")
        logger.info(f"最高置信度: {np.max(max_probs):.4f}")
        
        # 10. 错误分析
        logger.info("=== 错误分析 ===")
        errors = []
        for i, (true_label, pred_label_id) in enumerate(zip(y_test, y_pred)):
            pred_label = label_encoder.classes_[pred_label_id]
            if true_label != pred_label:
                errors.append({
                    'sample_id': i,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': max_probs[i]
                })
        
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
        result_df['predicted_label'] = [label_encoder.classes_[pred_id] for pred_id in y_pred]
        result_df['confidence'] = max_probs
        result_df['is_correct'] = [true_label == label_encoder.classes_[pred_id] 
                                  for true_label, pred_id in zip(y_test, y_pred)]
        
        output_path = os.path.join(PROCESSED_DATA_DIR, "fine_grained_test_results.parquet")
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
            'high_confidence_predictions': high_confidence,
            'average_confidence': float(np.mean(max_probs)),
            'confidence_std': float(np.std(max_probs)),
            'prediction_distribution': {label_encoder.classes_[k]: v for k, v in prediction_counts.items()},
            'error_count': len(errors),
            'error_types': dict(error_types) if errors else {},
            'true_label_distribution': dict(Counter(y_test)),
            'attack_types_supported': list(label_encoder.classes_)
        }
        
        report_path = os.path.join(PROCESSED_DATA_DIR, "fine_grained_test_report.json")
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
    logger.info("开始测试细粒度异常分类模型...")
    
    try:
        test_report = test_fine_grained_model()
        
        logger.info("=== 测试完成 ===")
        logger.info(f"总体准确率: {test_report['accuracy']:.4f}")
        logger.info(f"高置信度预测: {test_report['high_confidence_predictions']}")
        logger.info(f"支持的攻击类型: {test_report['attack_types_supported']}")
        
        if test_report['accuracy'] > 0.8:
            logger.info("✅ 细粒度异常分类功能表现良好")
        elif test_report['accuracy'] > 0.6:
            logger.info("⚠️ 细粒度异常分类功能需要改进")
        else:
            logger.info("❌ 细粒度异常分类功能需要重新训练")
            
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        raise

if __name__ == "__main__":
    main()
