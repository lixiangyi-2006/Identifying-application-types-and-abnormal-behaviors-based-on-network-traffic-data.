#!/usr/bin/env python3
"""
异常数据分类功能完整测试脚本
验证XGBoost模型是否能正确分类异常数据
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
from src.anomaly_detection.data_loader import NetworkDataLoader
from shared.config import MODEL_DIR, PROCESSED_DATA_DIR
from shared.utils import setup_logging

def create_realistic_test_data():
    """创建更真实的测试数据"""
    logger = setup_logging()
    logger.info("创建真实测试数据...")
    
    np.random.seed(42)  # 固定随机种子
    test_data = []
    
    # 1. 正常HTTP流量 (30个样本)
    for i in range(30):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(100, 200)}",
            'dst_ip': f"10.0.0.{np.random.randint(1, 10)}",
            'src_port': np.random.randint(49152, 65535),
            'dst_port': np.random.choice([80, 443, 8080]),
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(5),  # 正常流量持续时间较长
            'total_packets': np.random.poisson(20),       # 正常包数
            'total_bytes': np.random.poisson(2000),     # 正常字节数
            'avg_packet_size': np.random.normal(1000, 200),
            'packets_per_second': np.random.normal(4, 1),
            'bytes_per_second': np.random.normal(4000, 1000),
            'min_packet_size': np.random.normal(64, 10),
            'max_packet_size': np.random.normal(1500, 100),
            'std_packet_size': np.random.normal(200, 50),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 3600),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.25, 0.1),  # 正常包间隔
            'inter_packet_time_std': np.random.normal(0.1, 0.05),
            'src_to_dst_packets': np.random.poisson(10),
            'dst_to_src_packets': np.random.poisson(10),
            'src_to_dst_bytes': np.random.poisson(1000),
            'dst_to_src_bytes': np.random.poisson(1000),
            'tcp_flags': np.random.choice([2, 18, 16]),  # SYN, SYN+ACK, ACK
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0,
            'anomaly_type': 'normal'
        }
        test_data.append(flow)
    
    # 2. DDoS攻击流量 (25个样本) - 高频率、短持续时间
    for i in range(25):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(1, 100)}",
            'dst_ip': "10.0.0.1",  # 攻击目标固定
            'src_port': np.random.randint(1024, 65535),
            'dst_port': 80,
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(0.1),  # 攻击持续时间短
            'total_packets': np.random.poisson(1000),       # 大量包
            'total_bytes': np.random.poisson(100000),     # 大量字节
            'avg_packet_size': np.random.normal(100, 20),  # 小包
            'packets_per_second': np.random.normal(1000, 200),  # 高频率
            'bytes_per_second': np.random.normal(100000, 20000),
            'min_packet_size': np.random.normal(40, 5),
            'max_packet_size': np.random.normal(200, 50),
            'std_packet_size': np.random.normal(50, 10),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 300),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.001, 0.0005),  # 极短包间隔
            'inter_packet_time_std': np.random.normal(0.0005, 0.0002),
            'src_to_dst_packets': np.random.poisson(500),
            'dst_to_src_packets': np.random.poisson(500),
            'src_to_dst_bytes': np.random.poisson(50000),
            'dst_to_src_bytes': np.random.poisson(50000),
            'tcp_flags': 2,  # 主要是SYN包
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0,
            'anomaly_type': 'ddos'
        }
        test_data.append(flow)
    
    # 3. 端口扫描流量 (20个样本) - 单包、多目标
    for i in range(20):
        flow = {
            'src_ip': "192.168.1.100",  # 扫描源固定
            'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.randint(1, 65535),  # 随机目标端口
            'protocol': 'TCP',
            'flow_duration': np.random.exponential(0.01),  # 极短持续时间
            'total_packets': np.random.poisson(1),          # 通常只有1个包
            'total_bytes': np.random.poisson(60),          # 小字节数
            'avg_packet_size': np.random.normal(60, 10),
            'packets_per_second': np.random.normal(100, 50),  # 高频率扫描
            'bytes_per_second': np.random.normal(6000, 1000),
            'min_packet_size': np.random.normal(40, 5),
            'max_packet_size': np.random.normal(80, 10),
            'std_packet_size': np.random.normal(10, 5),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 600),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.01, 0.005),
            'inter_packet_time_std': np.random.normal(0.005, 0.002),
            'src_to_dst_packets': np.random.poisson(1),
            'dst_to_src_packets': 0,  # 扫描通常没有回复
            'src_to_dst_bytes': np.random.poisson(60),
            'dst_to_src_bytes': 0,
            'tcp_flags': 2,  # SYN包
            'udp_length': 0,
            'icmp_type': 0,
            'icmp_code': 0,
            'anomaly_type': 'port_scan'
        }
        test_data.append(flow)
    
    # 4. 边界情况测试 (10个样本) - 难以分类的数据
    for i in range(10):
        flow = {
            'src_ip': f"192.168.1.{np.random.randint(200, 254)}",
            'dst_ip': f"10.0.0.{np.random.randint(10, 50)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([22, 23, 25, 53, 110, 143]),
            'protocol': np.random.choice(['TCP', 'UDP']),
            'flow_duration': np.random.exponential(2),  # 中等持续时间
            'total_packets': np.random.poisson(50),       # 中等包数
            'total_bytes': np.random.poisson(5000),       # 中等字节数
            'avg_packet_size': np.random.normal(100, 50),  # 中等包大小
            'packets_per_second': np.random.normal(25, 10),  # 中等频率
            'bytes_per_second': np.random.normal(2500, 1000),
            'min_packet_size': np.random.normal(40, 10),
            'max_packet_size': np.random.normal(1000, 200),
            'std_packet_size': np.random.normal(100, 30),
            'flow_start_time': datetime.now().timestamp() - np.random.uniform(0, 1800),
            'flow_end_time': datetime.now().timestamp(),
            'inter_packet_time_mean': np.random.normal(0.04, 0.02),
            'inter_packet_time_std': np.random.normal(0.02, 0.01),
            'src_to_dst_packets': np.random.poisson(25),
            'dst_to_src_packets': np.random.poisson(25),
            'src_to_dst_bytes': np.random.poisson(2500),
            'dst_to_src_bytes': np.random.poisson(2500),
            'tcp_flags': np.random.randint(0, 256),
            'udp_length': np.random.randint(8, 1500),
            'icmp_type': np.random.randint(0, 16),
            'icmp_code': np.random.randint(0, 16),
            'anomaly_type': 'normal'  # 标记为正常，但特征可能模糊
        }
        test_data.append(flow)
    
    # 创建DataFrame
    df = pd.DataFrame(test_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存测试数据
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    test_path = os.path.join(PROCESSED_DATA_DIR, "realistic_test_data.parquet")
    df.to_parquet(test_path, index=False)
    
    logger.info(f"真实测试数据已保存到: {test_path}")
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"真实类别分布:")
    print(df['anomaly_type'].value_counts())
    
    return test_path, df

def test_anomaly_classification():
    """测试异常数据分类功能"""
    logger = setup_logging()
    logger.info("=== 开始异常数据分类测试 ===")
    
    try:
        # 1. 创建真实测试数据
        test_path, test_df = create_realistic_test_data()
        
        # 2. 查找模型
        model_files = []
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.startswith("xgboost_anomaly_model_") and file.endswith(".pkl"):
                    model_files.append(os.path.join(MODEL_DIR, file))
        
        if not model_files:
            raise FileNotFoundError("未找到XGBoost模型文件")
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"使用模型: {latest_model}")
        
        # 3. 创建检测器和数据加载器
        detector = TwoStageAnomalyDetector(latest_model)
        data_loader = NetworkDataLoader()
        
        # 4. 加载和预处理数据
        df = data_loader.load_from_file(test_path)
        df_processed = data_loader.preprocess_network_data(df)
        
        # 5. 准备特征
        feature_columns = detector.xgboost_model.feature_columns
        X = df_processed[feature_columns].values
        
        # 6. 执行预测
        logger.info("开始预测...")
        results = detector.batch_predict(X, confidence_threshold=0.5)
        
        # 7. 分析结果
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
        
        # 8. 计算准确率
        true_labels = test_df['anomaly_type'].tolist()
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
        
        # 9. 按类别分析准确率
        logger.info("=== 各类别准确率 ===")
        for true_label in ['normal', 'ddos', 'port_scan']:
            true_indices = [i for i, label in enumerate(true_labels) if label == true_label]
            if true_indices:
                correct_in_class = sum(1 for i in true_indices if results['labels'][i] == true_label)
                class_accuracy = correct_in_class / len(true_indices)
                logger.info(f"{true_label}: {correct_in_class}/{len(true_indices)} = {class_accuracy:.4f}")
        
        # 10. 置信度分析
        logger.info("=== 置信度分析 ===")
        confidences = results['confidences']
        logger.info(f"平均置信度: {np.mean(confidences):.4f}")
        logger.info(f"置信度标准差: {np.std(confidences):.4f}")
        logger.info(f"最低置信度: {np.min(confidences):.4f}")
        logger.info(f"最高置信度: {np.max(confidences):.4f}")
        
        # 11. 错误分析
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
        
        # 12. 保存详细结果
        result_df = df_processed.copy()
        result_df['predicted_label'] = results['labels']
        result_df['confidence'] = results['confidences']
        result_df['detection_path'] = results['detection_paths']
        result_df['is_correct'] = [r['is_correct'] for r in detailed_results]
        
        output_path = os.path.join(PROCESSED_DATA_DIR, "detailed_test_results.parquet")
        result_df.to_parquet(output_path, index=False)
        logger.info(f"详细测试结果已保存到: {output_path}")
        
        # 13. 生成测试报告
        test_report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_path': latest_model,
            'test_data_path': test_path,
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
            'error_types': dict(error_types) if errors else {}
        }
        
        report_path = os.path.join(PROCESSED_DATA_DIR, "test_report.json")
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
    logger.info("开始异常数据分类功能测试...")
    
    try:
        test_report = test_anomaly_classification()
        
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
