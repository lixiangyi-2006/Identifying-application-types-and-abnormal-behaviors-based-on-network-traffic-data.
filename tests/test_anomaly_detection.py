from src.anomaly_detection.data_loader import load_train_test
from shared.config import FEATURE_COLUMNS

def test_data_loading():
    """测试数据加载是否正常"""
    try:
        X_train, X_test, y_train, y_test = load_train_test()
        assert not X_train.empty, "训练集为空！"
        assert X_train.columns.tolist() == FEATURE_COLUMNS, "特征列名不一致！"
        print("数据加载测试通过！")
    except Exception as e:
        print(f"测试失败：{e}")

if __name__ == "__main__":
    test_data_loading()