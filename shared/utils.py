import logging
import pandas as pd

def init_logger(name):
    """初始化日志，所有模块共用"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def load_csv(file_path):
    """安全加载CSV，处理文件不存在错误"""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError(f"文件不存在：{file_path}，请联系数据处理同学确认")