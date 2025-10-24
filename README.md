# 网络异常检测系统

基于网络流量数据的异常检测和分类系统，采用两阶段检测架构。

## 项目概述

本系统实现了基于机器学习的网络异常检测，采用以下架构：

1. **第一阶段**：LightGBM模型进行初步异常检测（良性/异常）
2. **第二阶段**：XGBoost模型对异常数据进行分类和误判检测

### 主要特性

- 🚀 **两阶段检测**：LightGBM + XGBoost 组合检测
- 📊 **多类型异常检测**：DDoS、端口扫描、恶意软件等
- 🔧 **完整流水线**：数据预处理、特征工程、模型训练、推理服务
- 🌐 **RESTful API**：提供HTTP接口进行实时检测
- 📈 **可视化分析**：训练结果和检测结果可视化
- 🧪 **完整测试**：单元测试和集成测试

## 项目结构

```
anomaly_detection_model_thb/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 预处理数据
│   ├── models/                    # 训练好的模型
│   └── intermediate/              # 中间数据
├── src/                           # 源代码
│   ├── anomaly_detection/         # 异常检测模块
│   │   ├── model.py              # XGBoost模型
│   │   ├── trainer.py            # 模型训练器
│   │   ├── inferencer.py         # 两阶段检测器
│   │   ├── data_loader.py        # 数据加载器
│   │   └── main.py               # 主程序
│   └── api_service.py            # API服务
├── examples/                      # 示例脚本
│   ├── training_example.py       # 训练示例
│   └── inference_example.py      # 推理示例
├── tests/                         # 测试文件
│   └── test_anomaly_detection.py # 单元测试
├── config/                        # 配置文件
│   └── model_config.yaml         # 模型配置
├── shared/                        # 共享模块
│   ├── config.py                 # 配置管理
│   └── utils.py                  # 工具函数
├── requirements.txt               # 依赖包
└── README.md                     # 项目说明
```

## 安装和设置

### 1. 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 创建必要目录

```bash
mkdir -p data/{raw,processed,models,intermediate}
mkdir -p logs
```

## 快速开始

### 1. 训练模型

```bash
# 使用示例数据训练
python examples/training_example.py

# 使用自定义数据训练
python src/anomaly_detection/main.py --mode train --data your_data.parquet --model models/my_model.pkl
```

### 2. 模型推理

```bash
# 单样本预测
python examples/inference_example.py

# 批量预测
python src/anomaly_detection/main.py --mode predict --data test_data.parquet --model models/my_model.pkl --output results.parquet
```

### 3. 启动API服务

```bash
python src/api_service.py
```

API服务将在 `http://localhost:8000` 启动，可以访问 `http://localhost:8000/docs` 查看API文档。

## 使用方法

### 训练模型

```python
from src.anomaly_detection.trainer import AnomalyDetectionTrainer

# 创建训练器
trainer = AnomalyDetectionTrainer()

# 执行训练
results = trainer.full_training_pipeline(
    data_path="data/processed/training_data.parquet",
    target_column="anomaly_type"
)

print(f"模型保存路径: {results['model_path']}")
print(f"测试集准确率: {results['test_results']['metrics']['accuracy']:.4f}")
```

### 异常检测

```python
from src.anomaly_detection.inferencer import TwoStageAnomalyDetector

# 创建检测器
detector = TwoStageAnomalyDetector("models/xgboost_model.pkl")

# 单样本检测
result = detector.predict_single(features_array)
print(f"检测结果: {result['label']}")
print(f"置信度: {result['confidence']:.4f}")

# 批量检测
results = detector.batch_predict(features_array)
```

### API调用

```python
import requests

# 单样本预测
response = requests.post("http://localhost:8000/predict/single", json={
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.1",
    "src_port": 12345,
    "dst_port": 80,
    "protocol": "TCP",
    "flow_duration": 10.5,
    "total_packets": 100,
    "total_bytes": 10000,
    "avg_packet_size": 100.0
})

result = response.json()
print(f"预测结果: {result['label']}")
```

## 数据格式

### 输入数据格式

系统支持以下网络流量特征：

- **基础连接信息**：src_ip, dst_ip, src_port, dst_port, protocol
- **流量统计特征**：flow_duration, total_packets, total_bytes, avg_packet_size
- **速率特征**：packets_per_second, bytes_per_second
- **包大小统计**：min_packet_size, max_packet_size, std_packet_size
- **时间特征**：flow_start_time, flow_end_time, inter_packet_time_mean
- **方向特征**：src_to_dst_packets, dst_to_src_packets
- **协议特征**：tcp_flags, udp_length, icmp_type, icmp_code

### 标签格式

- **异常类型**：normal, ddos, port_scan, malware, botnet, intrusion
- **检测路径**：lightgbm_benign, xgboost_anomaly, xgboost_low_confidence, needs_review

## 配置说明

### 模型参数

在 `config/model_config.yaml` 中可以配置：

- XGBoost模型参数
- 数据预处理参数
- 特征工程参数
- 检测阈值
- API服务配置

### 环境变量

```bash
export MODEL_DIR="data/models"
export LOG_LEVEL="INFO"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

## 测试

运行单元测试：

```bash
pytest tests/ -v
```

运行特定测试：

```bash
pytest tests/test_anomaly_detection.py::TestXGBoostAnomalyClassifier -v
```

## 性能优化

### 模型优化

- 使用特征选择减少维度
- 调整XGBoost超参数
- 使用早停机制防止过拟合

### 推理优化

- 批量处理提高吞吐量
- 模型量化减少内存使用
- 异步API处理提高并发

## 部署建议

### 开发环境

```bash
python src/api_service.py
```

### 生产环境

```bash
# 使用Gunicorn部署
gunicorn src.api_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# 使用Docker部署
docker build -t anomaly-detection .
docker run -p 8000:8000 anomaly-detection
```

## 常见问题

### Q: 如何处理缺失值？

A: 系统自动处理缺失值，数值列使用中位数填充，分类列使用众数填充。

### Q: 如何调整检测阈值？

A: 在 `config/model_config.yaml` 中修改 `detection_thresholds` 部分。

### Q: 如何添加新的异常类型？

A: 在 `shared/config.py` 中的 `Labels.ANOMALY_TYPES` 添加新类型，并重新训练模型。

### Q: 如何处理大规模数据？

A: 使用批量处理，调整 `batch_size` 参数，考虑使用分布式训练。

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 实现两阶段检测架构
- 提供完整的训练和推理流水线
- 添加RESTful API服务
- 包含完整的测试和文档