#!/bin/bash

# 模型训练脚本

echo "开始训练异常检测模型..."

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <数据文件路径> [模型保存路径] [目标列名]"
    echo "示例: $0 data/processed/training_data.parquet models/my_model.pkl anomaly_type"
    exit 1
fi

DATA_PATH=$1
MODEL_PATH=${2:-"models/xgboost_anomaly_model_$(date +%Y%m%d_%H%M%S).pkl"}
TARGET_COLUMN=${3:-"anomaly_type"}

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 创建模型目录
mkdir -p $(dirname "$MODEL_PATH")

echo "数据文件: $DATA_PATH"
echo "模型保存路径: $MODEL_PATH"
echo "目标列: $TARGET_COLUMN"
echo ""

# 运行训练
python src/anomaly_detection/main.py \
    --mode train \
    --data "$DATA_PATH" \
    --model "$MODEL_PATH" \
    --target "$TARGET_COLUMN"

if [ $? -eq 0 ]; then
    echo "✅ 模型训练完成！"
    echo "模型已保存到: $MODEL_PATH"
else
    echo "❌ 模型训练失败"
    exit 1
fi
