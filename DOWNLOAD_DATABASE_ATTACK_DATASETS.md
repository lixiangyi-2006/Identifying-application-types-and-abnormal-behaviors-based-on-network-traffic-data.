# 数据库攻击数据集下载指南

## 目标目录
`D:\data1\database_attack_data`

## 推荐数据集下载步骤

### 1. CICIDS2017 (最推荐) ⭐
**网址**: https://www.unb.ca/cic/datasets/ids-2017.html

**下载步骤**:
1. 访问网站并注册账号
2. 填写研究用途申请
3. 下载完整数据集(约50GB)
4. 解压到 `D:\data1\cicids2017`

**包含内容**:
- SQL注入攻击
- XSS攻击  
- DDoS攻击
- 多种数据库相关攻击

---

### 2. UNSW-NB15 (备选)
**网址**: https://research.unsw.edu.au/projects/unsw-nb15-dataset

**下载步骤**:
1. 直接访问网站
2. 下载CSV格式文件
3. 保存到 `D:\data1\unsw-nb15`

**包含内容**:
- Exploits攻击
- Fuzzers攻击
- 数据库漏洞利用

---

### 3. GitHub SQL注入数据集

**搜索网址**: https://github.com/search?q=SQL+injection+dataset

**下载步骤**:
1. 搜索 "SQL injection dataset"
2. 选择 star 数>10 的仓库
3. 下载数据文件到 `D:\data1\github_sql_datasets`

**推荐的GitHub仓库**:
- https://github.com/youngwookim/awesome-hacking-lists
- https://github.com/security-cheatsheet/SQL-Injection-Payloads

---

### 4. Kaggle数据集
**网址**: https://www.kaggle.com/datasets?search=database+attack

**搜索关键词**:
- "SQL injection"
- "database attack"  
- "network intrusion"

---

### 5. OWASP资源
**WebGoat**: https://github.com/WebGoat/WebGoat
- 可生成SQL注入数据用于训练

**DVWA**: https://github.com/digininja/DVWA  
- 包含SQL注入测试场景

---

## 快速开始建议

**最简单的方法**: 
1. 先下载 **UNSW-NB15** (下载最简单)
2. 然后申请 **CICIDS2017** (最全面)
3. 最后从 **GitHub** 补充特定攻击样本

---

## 数据整合

下载完成后，我可以帮你:
1. 解析和格式化新下载的数据
2. 提取数据库攻击相关特征
3. 与现有模型数据合并
4. 重新训练模型以提高数据库攻击检测效果

---

## 当前问题

你的模型在数据库攻击检测上F1分数为0.58，需要更多数据库攻击样本来改进:
- 当前数据比例: 2.4% (5,282 samples)
- 目标比例: 15-20%
- 需要补充: 约30,000-40,000个数据库攻击样本

下载新数据后，运行 `merge_datasets.py` 来整合数据！
