import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import joblib

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """设置日志配置"""
    logger = logging.getLogger("anomaly_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None) -> None:
    """保存模型到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存模型
    joblib.dump(model, filepath)
    
    # 保存元数据
    if metadata:
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_model(filepath: str) -> tuple[Any, Optional[Dict]]:
    """加载模型和元数据"""
    # 加载模型
    model = joblib.load(filepath)
    
    # 尝试加载元数据
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return model, metadata

def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'parquet') -> None:
    """保存DataFrame到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_dataframe(filepath: str, format: str = 'parquet') -> pd.DataFrame:
    """从文件加载DataFrame"""
    if format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """计算分类指标"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
    """打印分类报告"""
    from sklearn.metrics import classification_report
    
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """验证DataFrame是否包含必需的列"""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame缺少必需的列: {missing_columns}")
    return True

def handle_missing_values(df: pd.DataFrame, method: str = "median") -> pd.DataFrame:
    """处理缺失值"""
    df_clean = df.copy()
    
    if method == "drop":
        df_clean = df_clean.dropna()
    elif method == "mean":
        df_clean = df_clean.fillna(df_clean.mean())
    elif method == "median":
        df_clean = df_clean.fillna(df_clean.median())
    elif method == "mode":
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    else:
        raise ValueError(f"不支持的缺失值处理方法: {method}")
    
    return df_clean

def normalize_features(df: pd.DataFrame, columns: List[str], method: str = "standard") -> pd.DataFrame:
    """特征归一化"""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    df_normalized = df.copy()
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    df_normalized[columns] = scaler.fit_transform(df[columns])
    
    return df_normalized, scaler

def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """确保指定列是数值类型"""
    df_converted = df.copy()
    
    for col in columns:
        if col in df_converted.columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
    
    return df_converted

def create_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """创建特征摘要统计"""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # 分类变量摘要
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
            'value_counts': df[col].value_counts().head(10).to_dict()
        }
    
    return summary