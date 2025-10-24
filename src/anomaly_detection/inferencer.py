import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from datetime import datetime

from .model import XGBoostAnomalyClassifier
from shared.config import Labels, DataConfig
from shared.utils import setup_logging, load_model

class TwoStageAnomalyDetector:
    """两阶段异常检测器"""
    
    def __init__(self, xgboost_model_path: str, lightgbm_model_path: Optional[str] = None):
        self.logger = setup_logging()
        self.xgboost_model = XGBoostAnomalyClassifier()
        self.lightgbm_model = None
        self.lightgbm_model_path = lightgbm_model_path
        
        # 加载XGBoost模型
        self._load_xgboost_model(xgboost_model_path)
        
        # 如果提供了LightGBM模型路径，加载它
        if lightgbm_model_path:
            self._load_lightgbm_model(lightgbm_model_path)
    
    def _load_xgboost_model(self, model_path: str) -> None:
        """加载XGBoost模型"""
        try:
            self.xgboost_model.load(model_path)
            self.logger.info("XGBoost模型加载成功")
        except Exception as e:
            self.logger.error(f"XGBoost模型加载失败: {e}")
            raise
    
    def _load_lightgbm_model(self, model_path: str) -> None:
        """加载LightGBM模型"""
        try:
            # 这里需要根据实际的LightGBM模型格式来调整
            # 假设LightGBM模型也使用joblib保存
            self.lightgbm_model, _ = load_model(model_path)
            self.logger.info("LightGBM模型加载成功")
        except Exception as e:
            self.logger.warning(f"LightGBM模型加载失败: {e}")
            self.lightgbm_model = None
    
    def predict_lightgbm(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用LightGBM进行初步检测"""
        if self.lightgbm_model is None:
            raise ValueError("LightGBM模型未加载")
        
        # 预测概率
        probabilities = self.lightgbm_model.predict_proba(X)
        predictions = self.lightgbm_model.predict(X)
        
        return predictions, probabilities
    
    def predict_xgboost(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用XGBoost进行异常分类"""
        if not self.xgboost_model.is_trained:
            raise ValueError("XGBoost模型未训练")
        
        predictions = self.xgboost_model.predict(X)
        probabilities = self.xgboost_model.predict_proba(X)
        
        return predictions, probabilities
    
    def two_stage_detection(self, X: np.ndarray, 
                           confidence_threshold: float = 0.5,
                           return_details: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """两阶段检测流程"""
        self.logger.info("开始两阶段异常检测...")
        
        results = {
            'predictions': [],
            'probabilities': [],
            'stage1_results': [],
            'stage2_results': [],
            'final_labels': [],
            'confidence_scores': [],
            'detection_path': []
        }
        
        # 第一阶段：LightGBM检测
        if self.lightgbm_model is not None:
            try:
                lightgbm_pred, lightgbm_proba = self.predict_lightgbm(X)
                results['stage1_results'] = lightgbm_pred.tolist()
                
                # 判断是否为良性流量
                benign_mask = lightgbm_pred == 0  # 假设0表示良性
                anomaly_mask = ~benign_mask
                
                self.logger.info(f"LightGBM检测结果: {np.sum(benign_mask)} 良性, {np.sum(anomaly_mask)} 异常")
                
            except Exception as e:
                self.logger.warning(f"LightGBM检测失败: {e}")
                # 如果LightGBM失败，所有数据都进入第二阶段
                benign_mask = np.zeros(len(X), dtype=bool)
                anomaly_mask = np.ones(len(X), dtype=bool)
                results['stage1_results'] = [1] * len(X)  # 假设都是异常
        else:
            # 如果没有LightGBM模型，所有数据都进入第二阶段
            self.logger.info("未提供LightGBM模型，所有数据进入XGBoost检测")
            benign_mask = np.zeros(len(X), dtype=bool)
            anomaly_mask = np.ones(len(X), dtype=bool)
            results['stage1_results'] = [1] * len(X)
        
        # 初始化最终结果
        final_predictions = np.full(len(X), -1, dtype=int)  # -1表示未分类
        final_probabilities = np.zeros((len(X), len(self.xgboost_model.label_encoder.classes_)))
        confidence_scores = np.zeros(len(X))
        detection_paths = ['unknown'] * len(X)
        
        # 处理良性流量
        if np.any(benign_mask):
            benign_indices = np.where(benign_mask)[0]
            final_predictions[benign_indices] = 0  # 假设0表示正常
            final_probabilities[benign_indices, 0] = 1.0  # 正常类别概率为1
            confidence_scores[benign_indices] = 1.0
            detection_paths = ['lightgbm_benign' if i in benign_indices else detection_paths[i] 
                             for i in range(len(X))]
        
        # 第二阶段：XGBoost分类异常流量
        if np.any(anomaly_mask):
            anomaly_indices = np.where(anomaly_mask)[0]
            anomaly_data = X[anomaly_indices]
            
            try:
                xgb_pred, xgb_proba = self.predict_xgboost(anomaly_data)
                results['stage2_results'] = xgb_pred.tolist()
                
                # 获取置信度
                max_proba = np.max(xgb_proba, axis=1)
                
                # 低置信度预测标记为需要人工审核
                low_confidence_mask = max_proba < confidence_threshold
                
                # 更新最终结果
                final_predictions[anomaly_indices] = xgb_pred
                final_probabilities[anomaly_indices] = xgb_proba
                confidence_scores[anomaly_indices] = max_proba
                
                # 标记低置信度预测
                low_conf_indices = anomaly_indices[low_confidence_mask]
                final_predictions[low_conf_indices] = -2  # -2表示需要人工审核
                
                # 更新检测路径
                for i, idx in enumerate(anomaly_indices):
                    if low_confidence_mask[i]:
                        detection_paths[idx] = 'xgboost_low_confidence'
                    else:
                        detection_paths[idx] = 'xgboost_anomaly'
                
                self.logger.info(f"XGBoost分类结果: {len(xgb_pred)} 个异常样本")
                self.logger.info(f"低置信度样本: {np.sum(low_confidence_mask)} 个")
                
            except Exception as e:
                self.logger.error(f"XGBoost分类失败: {e}")
                # 如果XGBoost失败，标记为需要人工审核
                final_predictions[anomaly_indices] = -3  # -3表示检测失败
                detection_paths = ['xgboost_failed' if i in anomaly_indices else detection_paths[i] 
                                 for i in range(len(X))]
        
        # 准备返回结果
        results['predictions'] = final_predictions.tolist()
        results['probabilities'] = final_probabilities.tolist()
        results['final_labels'] = self._convert_predictions_to_labels(final_predictions)
        results['confidence_scores'] = confidence_scores.tolist()
        results['detection_path'] = detection_paths
        
        self.logger.info("两阶段检测完成")
        
        if return_details:
            return results
        else:
            return final_predictions
    
    def _convert_predictions_to_labels(self, predictions: np.ndarray) -> List[str]:
        """将预测结果转换为标签"""
        labels = []
        for pred in predictions:
            if pred == -1:
                labels.append('unknown')
            elif pred == -2:
                labels.append('needs_review')
            elif pred == -3:
                labels.append('detection_failed')
            elif pred == 0:
                labels.append('normal')
            else:
                # 使用XGBoost的标签编码器
                if pred < len(self.xgboost_model.label_encoder.classes_):
                    labels.append(self.xgboost_model.label_encoder.classes_[pred])
                else:
                    labels.append('unknown')
        return labels
    
    def predict_single(self, features: Union[List, np.ndarray, pd.Series], 
                      confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """预测单个样本"""
        # 转换为numpy数组
        if isinstance(features, (list, pd.Series)):
            features = np.array(features).reshape(1, -1)
        elif isinstance(features, np.ndarray):
            features = features.reshape(1, -1)
        
        # 执行两阶段检测
        results = self.two_stage_detection(features, confidence_threshold, return_details=True)
        
        # 返回单个样本的结果
        return {
            'prediction': results['predictions'][0],
            'label': results['final_labels'][0],
            'confidence': results['confidence_scores'][0],
            'detection_path': results['detection_path'][0],
            'probabilities': results['probabilities'][0]
        }
    
    def batch_predict(self, X: np.ndarray, 
                     confidence_threshold: float = 0.5,
                     batch_size: int = 1000) -> Dict[str, Any]:
        """批量预测"""
        self.logger.info(f"开始批量预测，样本数量: {len(X)}")
        
        all_results = {
            'predictions': [],
            'labels': [],
            'confidences': [],
            'detection_paths': [],
            'probabilities': []
        }
        
        # 分批处理
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_results = self.two_stage_detection(batch_X, confidence_threshold, return_details=True)
            
            # 合并结果
            all_results['predictions'].extend(batch_results['predictions'])
            all_results['labels'].extend(batch_results['final_labels'])
            all_results['confidences'].extend(batch_results['confidence_scores'])
            all_results['detection_paths'].extend(batch_results['detection_path'])
            all_results['probabilities'].extend(batch_results['probabilities'])
        
        self.logger.info("批量预测完成")
        return all_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'xgboost_loaded': self.xgboost_model.is_trained,
            'lightgbm_loaded': self.lightgbm_model is not None,
            'xgboost_classes': self.xgboost_model.label_encoder.classes_.tolist() if self.xgboost_model.is_trained else [],
            'feature_count': len(self.xgboost_model.feature_columns) if self.xgboost_model.feature_columns else 0
        }
        return info
