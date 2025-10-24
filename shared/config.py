import os
from typing import List, Dict, Any

# 项目根路径（自动计算）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径配置
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(DATA_DIR, "models")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")

# 模型路径
LIGHTGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

# 日志配置
LOG_DIR = os.path.join(ROOT_DIR, "logs")
LOG_LEVEL = "INFO"

# 网络流量特征列名（与数据处理同学确认）
NETWORK_FEATURES = [
    # 基础连接信息
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    
    # 流量统计特征
    "flow_duration", "total_packets", "total_bytes", "avg_packet_size",
    "packets_per_second", "bytes_per_second",
    
    # 包大小统计
    "min_packet_size", "max_packet_size", "std_packet_size",
    
    # 时间特征
    "flow_start_time", "flow_end_time", "inter_packet_time_mean",
    "inter_packet_time_std",
    
    # 方向特征
    "src_to_dst_packets", "dst_to_src_packets", "src_to_dst_bytes", "dst_to_src_bytes",
    
    # 协议特征
    "tcp_flags", "udp_length", "icmp_type", "icmp_code"
]

# 标签定义
class Labels:
    # LightGBM二分类标签
    BINARY_LABELS = {
        "benign": 0,
        "anomaly": 1
    }
    
    # XGBoost多分类标签
    ANOMALY_TYPES = {
        "normal": 0,
        "ddos": 1,
        "port_scan": 2,
        "malware": 3,
        "botnet": 4,
        "intrusion": 5
    }
    
    # 应用类型标签（用于应用识别）
    APP_TYPES = {
        "HTTP": 0,
        "FTP": 1,
        "DNS": 2,
        "P2P": 3,
        "SSH": 4,
        "SMTP": 5,
        "POP3": 6,
        "IMAP": 7
    }

# 模型配置
class ModelConfig:
    # XGBoost参数
    XGBOOST_PARAMS = {
        "objective": "multi:softprob",
        "num_class": len(Labels.ANOMALY_TYPES),
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    # LightGBM参数（用于参考）
    LIGHTGBM_PARAMS = {
        "objective": "binary",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

# 数据预处理配置
class DataConfig:
    # 数据分割比例
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 特征工程配置
    FEATURE_SCALING = True
    HANDLE_MISSING = "median"  # "mean", "median", "mode", "drop"
    
    # 异常检测阈值
    ANOMALY_THRESHOLD = 0.5

# API配置
class APIConfig:
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    TITLE = "Network Anomaly Detection API"
    VERSION = "1.0.0"

# 确保目录存在
def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, 
        INTERMEDIATE_DIR, LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 初始化时创建目录
ensure_directories()