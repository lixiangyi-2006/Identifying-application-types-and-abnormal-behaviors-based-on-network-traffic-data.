import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os

from shared.config import ModelConfig, Labels, DataConfig
from shared.utils import setup_logging, save_model, load_model, calculate_metrics

class XGBoostAnomalyClassifier:
    """XGBoost异常分类模型"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = setup_logging()
        self.config = config or ModelConfig.XGBOOST_PARAMS
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'anomaly_type') -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        self.logger.info("准备训练数据...")
        
        # 确保目标列存在
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
        
        # 分离特征和目标
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 保存特征列名
        self.feature_columns = list(X.columns)
        
        # 编码目标变量
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"数据形状: {X.shape}")
        self.logger.info(f"目标类别: {self.label_encoder.classes_}")
        
        return X.values, y_encoded
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2, 
              early_stopping_rounds: int = 10) -> Dict[str, Any]:
        """训练模型"""
        self.logger.info("开始训练XGBoost模型...")
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # 创建XGBoost模型
        self.model = xgb.XGBClassifier(**self.config)
        
        # 训练模型
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # 预测和评估
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)
        
        # 计算指标
        metrics = calculate_metrics(y_val, y_pred, self.label_encoder.classes_)
        
        self.logger.info(f"训练完成! 验证集准确率: {metrics['accuracy']:.4f}")
        self.logger.info(f"F1分数 (macro): {metrics['f1_macro']:.4f}")
        
        self.is_trained = True
        
        return {
            'metrics': metrics,
            'feature_importance': self.get_feature_importance(),
            'best_iteration': self.model.get_booster().best_iteration
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """带置信度的预测"""
        probabilities = self.predict_proba(X)
        predictions = self.model.predict(X)
        max_probs = np.max(probabilities, axis=1)
        
        # 低置信度预测标记为-1
        low_confidence_mask = max_probs < confidence_threshold
        predictions[low_confidence_mask] = -1
        
        return predictions, max_probs
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if self.feature_columns is None:
            return {}
        
        importance_scores = self.model.get_booster().get_score(importance_type=importance_type)
        
        # 转换为特征名
        feature_importance = {}
        for i, feature in enumerate(self.feature_columns):
            feature_importance[feature] = importance_scores.get(f'f{i}', 0.0)
        
        # 按重要性排序
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        # 准备保存的数据
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'config': self.config
        }
        
        # 保存元数据
        metadata = {
            'model_type': 'XGBoostAnomalyClassifier',
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'class_count': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        save_model(model_data, filepath, metadata)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        model_data, metadata = load_model(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data['config']
        self.is_trained = True
        
        self.logger.info(f"模型已从 {filepath} 加载")
        self.logger.info(f"特征数量: {len(self.feature_columns)}")
        self.logger.info(f"类别: {self.label_encoder.classes_}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """评估模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # 计算指标
        metrics = calculate_metrics(y, y_pred, self.label_encoder.classes_)
        
        # 分类报告
        report = classification_report(
            y, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': self.get_feature_importance()
        }
