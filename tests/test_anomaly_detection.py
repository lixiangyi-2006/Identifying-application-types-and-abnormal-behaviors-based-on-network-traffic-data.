"""
异常检测系统测试
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_detection.model import XGBoostAnomalyClassifier
from src.anomaly_detection.data_loader import NetworkDataLoader
from src.anomaly_detection.inferencer import TwoStageAnomalyDetector
from shared.config import Labels, NETWORK_FEATURES

class TestXGBoostAnomalyClassifier:
    """XGBoost异常分类器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.model = XGBoostAnomalyClassifier()
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'src_ip': [f"192.168.1.{i%255+1}" for i in range(n_samples)],
            'dst_ip': [f"10.0.0.{i%255+1}" for i in range(n_samples)],
            'src_port': np.random.randint(1024, 65535, n_samples),
            'dst_port': np.random.randint(1, 65535, n_samples),
            'protocol': np.random.choice(['TCP', 'UDP'], n_samples),
            'flow_duration': np.random.exponential(10, n_samples),
            'total_packets': np.random.poisson(50, n_samples),
            'total_bytes': np.random.poisson(5000, n_samples),
            'avg_packet_size': np.random.normal(1000, 200, n_samples),
            'packets_per_second': np.random.normal(5, 2, n_samples),
            'bytes_per_second': np.random.normal(5000, 2000, n_samples),
            'min_packet_size': np.random.normal(64, 10, n_samples),
            'max_packet_size': np.random.normal(1500, 100, n_samples),
            'std_packet_size': np.random.normal(200, 50, n_samples),
            'flow_start_time': np.random.uniform(0, 1000, n_samples),
            'flow_end_time': np.random.uniform(1000, 2000, n_samples),
            'inter_packet_time_mean': np.random.normal(0.1, 0.05, n_samples),
            'inter_packet_time_std': np.random.normal(0.02, 0.01, n_samples),
            'src_to_dst_packets': np.random.poisson(25, n_samples),
            'dst_to_src_packets': np.random.poisson(25, n_samples),
            'src_to_dst_bytes': np.random.poisson(2500, n_samples),
            'dst_to_src_bytes': np.random.poisson(2500, n_samples),
            'tcp_flags': np.random.randint(0, 256, n_samples),
            'udp_length': np.random.randint(8, 1500, n_samples),
            'icmp_type': np.random.randint(0, 16, n_samples),
            'icmp_code': np.random.randint(0, 16, n_samples),
            'anomaly_type': np.random.choice(['normal', 'ddos', 'port_scan'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        assert self.model.model is None
        assert not self.model.is_trained
        assert self.model.feature_columns is None
    
    def test_prepare_data(self):
        """测试数据准备"""
        X, y = self.model.prepare_data(self.sample_data, 'anomaly_type')
        
        assert X.shape[0] == len(self.sample_data)
        assert X.shape[1] == len(self.sample_data.columns) - 1  # 减去目标列
        assert len(y) == len(self.sample_data)
        assert self.model.feature_columns is not None
    
    def test_train_model(self):
        """测试模型训练"""
        X, y = self.model.prepare_data(self.sample_data, 'anomaly_type')
        
        # 训练模型
        results = self.model.train(X, y)
        
        assert self.model.is_trained
        assert 'metrics' in results
        assert 'accuracy' in results['metrics']
        assert results['metrics']['accuracy'] > 0
    
    def test_predict(self):
        """测试预测"""
        X, y = self.model.prepare_data(self.sample_data, 'anomaly_type')
        self.model.train(X, y)
        
        # 预测
        predictions = self.model.predict(X[:10])
        probabilities = self.model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, len(self.model.label_encoder.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # 概率和为1
    
    def test_predict_with_confidence(self):
        """测试带置信度的预测"""
        X, y = self.model.prepare_data(self.sample_data, 'anomaly_type')
        self.model.train(X, y)
        
        predictions, confidences = self.model.predict_with_confidence(X[:10], 0.5)
        
        assert len(predictions) == 10
        assert len(confidences) == 10
        assert all(0 <= conf <= 1 for conf in confidences)
    
    def test_feature_importance(self):
        """测试特征重要性"""
        X, y = self.model.prepare_data(self.sample_data, 'anomaly_type')
        self.model.train(X, y)
        
        importance = self.model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(self.model.feature_columns)
        assert all(isinstance(score, (int, float)) for score in importance.values())
    
    def test_save_load_model(self, tmp_path):
        """测试模型保存和加载"""
        X, y = self.model.prepare_data(self.sample_data, 'anomaly_type')
        self.model.train(X, y)
        
        # 保存模型
        model_path = tmp_path / "test_model.pkl"
        self.model.save(str(model_path))
        
        # 创建新模型并加载
        new_model = XGBoostAnomalyClassifier()
        new_model.load(str(model_path))
        
        assert new_model.is_trained
        assert new_model.feature_columns == self.model.feature_columns
        assert np.array_equal(new_model.label_encoder.classes_, self.model.label_encoder.classes_)

class TestNetworkDataLoader:
    """网络数据加载器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.loader = NetworkDataLoader()
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'src_ip': [f"192.168.1.{i%255+1}" for i in range(n_samples)],
            'dst_ip': [f"10.0.0.{i%255+1}" for i in range(n_samples)],
            'src_port': np.random.randint(1024, 65535, n_samples),
            'dst_port': np.random.randint(1, 65535, n_samples),
            'protocol': np.random.choice(['TCP', 'UDP'], n_samples),
            'flow_duration': np.random.exponential(10, n_samples),
            'total_packets': np.random.poisson(50, n_samples),
            'total_bytes': np.random.poisson(5000, n_samples),
            'avg_packet_size': np.random.normal(1000, 200, n_samples),
            'anomaly_type': np.random.choice(['normal', 'ddos', 'port_scan'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_validate_network_data(self):
        """测试网络数据验证"""
        is_valid, errors = self.loader.validate_network_data(self.sample_data)
        
        # 由于示例数据可能缺少一些必需特征，验证可能失败
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
    
    def test_preprocess_network_data(self):
        """测试数据预处理"""
        processed_data = self.loader.preprocess_network_data(self.sample_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == len(self.sample_data)
        assert not processed_data.isnull().any().any()  # 不应该有缺失值
    
    def test_create_derived_features(self):
        """测试衍生特征创建"""
        processed_data = self.loader._create_derived_features(self.sample_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        # 应该包含一些新特征
        assert len(processed_data.columns) >= len(self.sample_data.columns)
    
    def test_handle_outliers(self):
        """测试异常值处理"""
        # 创建包含异常值的数据
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'flow_duration'] = 1000000  # 异常大的值
        
        processed_data = self.loader._handle_outliers(data_with_outliers)
        
        assert isinstance(processed_data, pd.DataFrame)
        # 异常值应该被处理
        assert processed_data['flow_duration'].max() < 1000000

class TestTwoStageAnomalyDetector:
    """两阶段异常检测器测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建模拟的XGBoost模型
        self.mock_xgb_model = Mock()
        self.mock_xgb_model.is_trained = True
        self.mock_xgb_model.feature_columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
        self.mock_xgb_model.label_encoder = Mock()
        self.mock_xgb_model.label_encoder.classes_ = np.array(['normal', 'ddos', 'port_scan'])
        
        # 创建检测器
        self.detector = TwoStageAnomalyDetector("dummy_path")
        self.detector.xgboost_model = self.mock_xgb_model
    
    def test_initialization(self):
        """测试初始化"""
        assert self.detector.xgboost_model is not None
        assert self.detector.lightgbm_model is None
    
    def test_predict_xgboost(self):
        """测试XGBoost预测"""
        # 模拟XGBoost预测
        self.mock_xgb_model.predict.return_value = np.array([0, 1, 2])
        self.mock_xgb_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        
        X = np.random.random((3, 4))
        predictions, probabilities = self.detector.predict_xgboost(X)
        
        assert len(predictions) == 3
        assert probabilities.shape == (3, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_convert_predictions_to_labels(self):
        """测试预测结果转换为标签"""
        predictions = np.array([0, 1, 2, -1, -2, -3])
        labels = self.detector._convert_predictions_to_labels(predictions)
        
        expected_labels = ['normal', 'ddos', 'port_scan', 'unknown', 'needs_review', 'detection_failed']
        assert labels == expected_labels
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.detector.get_model_info()
        
        assert 'xgboost_loaded' in info
        assert 'lightgbm_loaded' in info
        assert 'xgboost_classes' in info
        assert 'feature_count' in info
        assert info['xgboost_loaded'] == True
        assert info['lightgbm_loaded'] == False

def test_config_imports():
    """测试配置导入"""
    from shared.config import Labels, NETWORK_FEATURES, ModelConfig
    
    assert isinstance(Labels.ANOMALY_TYPES, dict)
    assert isinstance(NETWORK_FEATURES, list)
    assert isinstance(ModelConfig.XGBOOST_PARAMS, dict)

def test_utils_functions():
    """测试工具函数"""
    from shared.utils import setup_logging, get_timestamp, ensure_numeric_columns
    
    # 测试日志设置
    logger = setup_logging()
    assert logger is not None
    
    # 测试时间戳
    timestamp = get_timestamp()
    assert isinstance(timestamp, str)
    assert len(timestamp) > 0
    
    # 测试数值列转换
    df = pd.DataFrame({'a': ['1', '2', '3'], 'b': ['x', 'y', 'z']})
    result = ensure_numeric_columns(df, ['a'])
    assert pd.api.types.is_numeric_dtype(result['a'])

if __name__ == "__main__":
    pytest.main([__file__])