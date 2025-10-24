#!/bin/bash

# 网络异常检测系统安装脚本

echo "开始安装网络异常检测系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

echo "Python版本检查通过: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 创建必要目录
echo "创建项目目录..."
mkdir -p data/{raw,processed,models,intermediate}
mkdir -p logs
mkdir -p config

# 设置权限
echo "设置文件权限..."
chmod +x src/anomaly_detection/main.py
chmod +x examples/*.py
chmod +x scripts/*.sh

# 运行测试
echo "运行测试..."
python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✅ 安装完成！"
    echo ""
    echo "使用方法："
    echo "1. 激活虚拟环境: source venv/bin/activate"
    echo "2. 训练模型: python examples/training_example.py"
    echo "3. 运行推理: python examples/inference_example.py"
    echo "4. 启动API: python src/api_service.py"
    echo ""
    echo "更多信息请查看 README.md"
else
    echo "❌ 测试失败，请检查安装"
    exit 1
fi
