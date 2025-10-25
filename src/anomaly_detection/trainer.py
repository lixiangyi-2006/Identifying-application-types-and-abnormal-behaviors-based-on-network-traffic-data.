import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split

from .model import XGBoostAnomalyClassifier
from shared.config import DataConfig, Labels, MODEL_DIR
from shared.utils import setup_logging, save_dataframe, load_dataframe, print_classification_report

class AnomalyDetectionTrainer:
    """异常检测模型训练器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = setup_logging()
        self.config = config or {}
        self.model = XGBoostAnomalyClassifier()
        self.training_history = []
        
    def load_data(self, data_path: str, format: str = 'parquet') -> pd.DataFrame:
        """加载训练数据"""
        self.logger.info(f"从 {data_path} 加载数据...")
        
        try:
            df = load_dataframe(data_path, format)
            self.logger.info(f"数据加载成功，形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'anomaly_type') -> pd.DataFrame:
        """数据预处理"""
        self.logger.info("开始数据预处理...")
        
        # 使用NetworkDataLoader进行预处理
        from .data_loader import NetworkDataLoader
        data_loader = NetworkDataLoader()
        df_processed = data_loader.preprocess_network_data(df, target_column)
        
        # 处理目标变量
        if target_column in df_processed.columns:
            # 检查目标变量的分布
            target_distribution = df_processed[target_column].value_counts()
            self.logger.info(f"目标变量分布:\n{target_distribution}")
            
            # 检查是否有未知类别
            known_labels = set(Labels.ANOMALY_TYPES.keys())
            data_labels = set(df_processed[target_column].unique())
            unknown_labels = data_labels - known_labels
            
            if unknown_labels:
                self.logger.warning(f"发现未知标签: {unknown_labels}")
                # 将未知标签映射为 'normal'
                df_processed[target_column] = df_processed[target_column].replace(
                    list(unknown_labels), 'normal'
                )
        
        self.logger.info(f"预处理完成，最终形状: {df_processed.shape}")
        return df_processed
    
    def split_data(self, df: pd.DataFrame, target_column: str = 'anomaly_type') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """分割数据集"""
        self.logger.info("分割数据集...")
        
        # 分层抽样确保各类别比例一致
        train_df, temp_df = train_test_split(
            df, 
            test_size=1 - DataConfig.TRAIN_RATIO,
            random_state=42,
            stratify=df[target_column] if target_column in df.columns else None
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=DataConfig.TEST_RATIO / (DataConfig.VALIDATION_RATIO + DataConfig.TEST_RATIO),
            random_state=42,
            stratify=temp_df[target_column] if target_column in temp_df.columns else None
        )
        
        self.logger.info(f"训练集: {train_df.shape}")
        self.logger.info(f"验证集: {val_df.shape}")
        self.logger.info(f"测试集: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   target_column: str = 'anomaly_type') -> Dict[str, Any]:
        """训练模型"""
        self.logger.info("开始训练模型...")
        
        # 准备训练数据
        X_train, y_train = self.model.prepare_data(train_df, target_column)
        X_val, y_val = self.model.prepare_data(val_df, target_column)
        
        # 训练模型
        training_results = self.model.train(X_train, y_train, validation_split=0.2)
        
        # 在验证集上评估
        val_results = self.model.evaluate(X_val, y_val)
        
        # 保存训练历史
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_metrics': training_results['metrics'],
            'val_metrics': val_results['metrics'],
            'feature_importance': training_results['feature_importance']
        }
        self.training_history.append(training_record)
        
        self.logger.info("模型训练完成!")
        self.logger.info(f"验证集准确率: {val_results['metrics']['accuracy']:.4f}")
        
        return {
            'training_results': training_results,
            'validation_results': val_results,
            'training_history': training_record
        }
    
    def evaluate_model(self, test_df: pd.DataFrame, target_column: str = 'anomaly_type') -> Dict[str, Any]:
        """在测试集上评估模型"""
        self.logger.info("在测试集上评估模型...")
        
        # 准备测试数据
        X_test, y_test = self.model.prepare_data(test_df, target_column)
        
        # 评估模型
        test_results = self.model.evaluate(X_test, y_test)
        
        # 打印详细报告
        y_pred = self.model.predict(X_test)
        print_classification_report(y_test, y_pred, self.model.label_encoder.classes_)
        
        self.logger.info(f"测试集准确率: {test_results['metrics']['accuracy']:.4f}")
        
        return test_results
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """保存训练好的模型"""
        if not self.model.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(MODEL_DIR, f"xgboost_anomaly_model_{timestamp}.pkl")
        
        self.model.save(model_path)
        self.logger.info(f"模型已保存到: {model_path}")
        
        return model_path
    
    def plot_training_results(self, save_path: Optional[str] = None) -> None:
        """绘制训练结果"""
        if not self.training_history:
            self.logger.warning("没有训练历史数据可绘制")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('模型训练结果', fontsize=16)
        
        # 1. 特征重要性
        if 'feature_importance' in self.training_history[-1]:
            importance = self.training_history[-1]['feature_importance']
            top_features = dict(list(importance.items())[:10])
            
            axes[0, 0].barh(list(top_features.keys()), list(top_features.values()))
            axes[0, 0].set_title('Top 10 特征重要性')
            axes[0, 0].set_xlabel('重要性分数')
        
        # 2. 训练指标趋势（如果有多次训练）
        if len(self.training_history) > 1:
            timestamps = [record['timestamp'] for record in self.training_history]
            train_acc = [record['train_metrics']['accuracy'] for record in self.training_history]
            val_acc = [record['val_metrics']['accuracy'] for record in self.training_history]
            
            axes[0, 1].plot(range(len(timestamps)), train_acc, label='训练准确率', marker='o')
            axes[0, 1].plot(range(len(timestamps)), val_acc, label='验证准确率', marker='s')
            axes[0, 1].set_title('准确率趋势')
            axes[0, 1].set_xlabel('训练轮次')
            axes[0, 1].set_ylabel('准确率')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 3. 类别分布
        if 'val_metrics' in self.training_history[-1]:
            metrics = self.training_history[-1]['val_metrics']
            metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            metric_values = [metrics[name] for name in metric_names]
            
            axes[1, 0].bar(metric_names, metric_values)
            axes[1, 0].set_title('验证集指标')
            axes[1, 0].set_ylabel('分数')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 混淆矩阵（如果有测试结果）
        axes[1, 1].text(0.5, 0.5, '混淆矩阵\n(需要测试数据)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('混淆矩阵')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"训练结果图表已保存到: {save_path}")
        
        plt.show()
    
    def full_training_pipeline(self, data_path: str, target_column: str = 'anomaly_type',
                             save_model_path: Optional[str] = None) -> Dict[str, Any]:
        """完整的训练流水线"""
        self.logger.info("开始完整训练流水线...")
        
        try:
            # 1. 加载数据
            df = self.load_data(data_path)
            
            # 2. 数据预处理
            df_processed = self.preprocess_data(df, target_column)
            
            # 3. 分割数据
            train_df, val_df, test_df = self.split_data(df_processed, target_column)
            
            # 4. 训练模型
            training_results = self.train_model(train_df, val_df, target_column)
            
            # 5. 测试评估
            test_results = self.evaluate_model(test_df, target_column)
            
            # 6. 保存模型
            model_path = self.save_model(save_model_path)
            
            # 7. 绘制结果
            plot_path = model_path.replace('.pkl', '_training_results.png')
            self.plot_training_results(plot_path)
            
            self.logger.info("完整训练流水线执行完成!")
            
            return {
                'training_results': training_results,
                'test_results': test_results,
                'model_path': model_path,
                'plot_path': plot_path
            }
            
        except Exception as e:
            self.logger.error(f"训练流水线执行失败: {e}")
            raise
