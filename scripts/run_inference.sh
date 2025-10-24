#!/bin/bash

# 模型推理脚本

echo "开始异常检测推理..."

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法: $0 <模型路径> <数据文件路径> [输出路径] [LightGBM模型路径]"
    echo "示例: $0 models/xgboost_model.pkl data/test_data.parquet results.parquet models/lightgbm_model.pkl"
    exit 1
fi

MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_PATH=${3:-"results_$(date +%Y%m%d_%H%M%S).parquet"}
LIGHTGBM_MODEL_PATH=${4:-""}

# 检查文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

echo "模型文件: $MODEL_PATH"
echo "数据文件: $DATA_PATH"
echo "输出文件: $OUTPUT_PATH"
if [ -n "$LIGHTGBM_MODEL_PATH" ]; then
    echo "LightGBM模型: $LIGHTGBM_MODEL_PATH"
fi
echo ""

# 运行推理
if [ -n "$LIGHTGBM_MODEL_PATH" ]; then
    python src/anomaly_detection/main.py \
        --mode predict \
        --model "$MODEL_PATH" \
        --data "$DATA_PATH" \
        --output "$OUTPUT_PATH" \
        --lightgbm-model "$LIGHTGBM_MODEL_PATH"
else
    python src/anomaly_detection/main.py \
        --mode predict \
        --model "$MODEL_PATH" \
        --data "$DATA_PATH" \
        --output "$OUTPUT_PATH"
fi

if [ $? -eq 0 ]; then
    echo "✅ 推理完成！"
    echo "结果已保存到: $OUTPUT_PATH"
else
    echo "❌ 推理失败"
    exit 1
fi
