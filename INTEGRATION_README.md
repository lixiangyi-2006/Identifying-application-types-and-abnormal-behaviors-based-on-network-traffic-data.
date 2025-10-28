# 数据集整合说明

## 快速整合脚本

我已经创建了两个脚本用于整合 `D:\data1` 中的数据集：

### 方式1: 使用 quick_merge.py (推荐，更简单)

直接在PowerShell或命令提示符中运行：

```bash
python quick_merge.py
```

这个脚本会：
1. 扫描 `D:\data1` 目录中的所有Excel文件
2. 根据文件名自动识别标签类型：
   - 良性/正常 → normal
   - 暴力破解 → brute_force
   - 欺骗 → spoofing
   - 上传危机 → upload_attack
   - 数据库攻击 → database_attack
3. 合并所有数据
4. 处理数据质量问题（无穷大值、缺失值）
5. 按80/20比例划分训练/测试集（使用分层抽样）
6. 保存到 `data/processed/train.xlsx` 和 `data/processed/test.xlsx`

### 方式2: 使用 merge_datasets.py (更详细)

```bash
python merge_datasets.py
```

这个脚本功能相同，但包含更详细的日志和统计信息。

## 输出文件

整合完成后，会在 `data/processed/` 目录下生成：
- `train.xlsx` - 训练集
- `test.xlsx` - 测试集
- `dataset_statistics.json` - 数据统计信息（使用merge_datasets.py时）

## 注意事项

1. **确保数据目录存在**: `D:\data1` 目录中应包含5个Excel文件
2. **文件名识别**: 脚本通过文件名关键词识别数据类型，确保文件名包含相应关键词
3. **数据比例**: 如果某个类别样本过多，可以考虑在脚本中启用平衡采样

## 整合后的下一步

整合完成后，可以：
1. 运行 `python train_balanced_model.py` 训练模型
2. 或运行 `python fix_data_quality.py` 进一步处理数据质量
3. 或直接使用预处理后的数据进行训练
