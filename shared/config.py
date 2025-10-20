import os

# 项目根路径（自动计算）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据路径
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_APP_DIR = os.path.join(DATA_DIR, "processed", "app_recognition")
PROCESSED_ANOMALY_DIR = os.path.join(DATA_DIR, "processed", "anomaly_detection")

# 共享特征列名（与数据处理同学确认）
FEATURE_COLUMNS = [
    "src_ip", "dst_ip", "src_port", "dst_port",
    "flow_duration", "total_packets", "avg_packet_size"
]

# 标签定义
APP_TYPES = ["HTTP", "FTP", "DNS", "P2P"]  # 应用识别模块用
ANOMALY_TYPES = ["normal", "DDoS", "port_scan"]  # 你的模块用