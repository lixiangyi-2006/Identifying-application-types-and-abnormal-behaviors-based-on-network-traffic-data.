import os
from shared.config import PROCESSED_ANOMALY_DIR, FEATURE_COLUMNS
from shared.utils import load_csv, init_logger

logger = init_logger("anomaly_data_loader")


def load_train_test():
    """加载训练集和测试集"""
    train_path = os.path.join(PROCESSED_ANOMALY_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_ANOMALY_DIR, "test.csv")

    logger.info(f"从 {train_path} 加载训练集...")
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    # 提取特征和标签
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["is_anomaly"]  # 标签列名与数据处理同学确认
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["is_anomaly"]

    return X_train, X_test, y_train, y_test