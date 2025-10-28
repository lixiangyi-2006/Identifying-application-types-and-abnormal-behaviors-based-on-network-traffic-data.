#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库攻击数据集资源指南
提供获取数据库攻击相关数据集的途径和建议
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_dataset_sources():
    """打印数据库攻击数据集资源"""
    
    print("=" * 80)
    print("数据库攻击数据集资源指南")
    print("=" * 80)
    
    print("\n1. 公开的网络安全数据集（包含SQL注入、数据库攻击）")
    print("-" * 80)
    
    sources = [
        {
            'name': 'CICIDS2017',
            'description': '包含多种网络入侵类型，包括数据库攻击',
            'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
            'features': ['包含多类攻击', '真实网络流量', '标注完整'],
            'download': '需要注册访问'
        },
        {
            'name': 'UNSW-NB15',
            'description': '包含9类攻击，包含数据库相关漏洞利用',
            'url': 'https://research.unsw.edu.au/projects/unsw-nb15-dataset',
            'features': ['流量丰富', '攻击类型多样', '研究广泛使用'],
            'download': '直接下载'
        },
        {
            'name': 'NSL-KDD',
            'description': '经典的入侵检测数据集',
            'url': 'http://www.unb.ca/cic/datasets/nsl.html',
            'features': ['包含后端攻击', '基准数据集', '轻量级'],
            'download': '直接下载'
        },
        {
            'name': 'CIC-DDoS2019',
            'description': '专门针对DDoS攻击，但包含数据库攻击流量',
            'url': 'https://www.unb.ca/cic/datasets/ddos-2019.html',
            'features': ['大规模数据', '真实环境', '性能评估'],
            'download': '需要注册'
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   描述: {source['description']}")
        print(f"   网址: {source['url']}")
        print(f"   特点: {', '.join(source['features'])}")
        print(f"   下载: {source['download']}")
    
    print("\n" + "=" * 80)
    print("2. 专业SQL注入和数据库攻击数据集")
    print("-" * 80)
    
    sql_sources = [
        {
            'name': 'SQL Injection Dataset',
            'description': '专门针对SQL注入攻击的数据集',
            'url': 'https://github.com/PacktPublishing/Machine-Learning-for-Cybersecurity-Cookbook',
            'features': ['SQL注入样本', '多种注入技术', 'HTTP请求数据']
        },
        {
            'name': 'OWASP WebGoat',
            'description': 'OWASP提供的安全练习应用，可生成数据库攻击数据',
            'url': 'https://github.com/WebGoat/WebGoat',
            'features': ['可生成数据', '包含SQL注入', 'NoSQL注入场景']
        },
        {
            'name': 'DVWA (Damn Vulnerable Web Application)',
            'description': '包含漏洞的Web应用，可以生成SQL注入攻击数据',
            'url': 'https://github.com/digininja/DVWA',
            'features': ['SQL注入场景', 'MySQL注入', 'SQL盲注']
        }
    ]
    
    for i, source in enumerate(sql_sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   描述: {source['description']}")
        print(f"   网址: {source['url']}")
        print(f"   特点: {', '.join(source['features'])}")
    
    print("\n" + "=" * 80)
    print("3. 机器学习竞赛数据集")
    print("-" * 80)
    
    competition_sources = [
        {
            'name': 'Kaggle - Network Security Datasets',
            'description': '多个网络安全相关的竞赛数据集',
            'url': 'https://www.kaggle.com/datasets?search=network+security',
            'features': ['SQL注入数据', '竞赛数据', '社区支持']
        },
        {
            'name': 'GitHub - Awesome Cyber Security Datasets',
            'description': '网络安全数据集集合',
            'url': 'https://github.com/AppliedML/Awesome-Cybersecurity-Datasets',
            'features': ['数据库攻击', '综合资源', '持续更新']
        },
        {
            'name': 'AI4Cyber Dataset List',
            'description': 'AI for Cybersecurity 数据集列表',
            'url': 'https://www.ai4cyber.com/datasets',
            'features': ['数据库漏洞', '入侵检测', '规范标注']
        }
    ]
    
    for i, source in enumerate(competition_sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   描述: {source['description']}")
        print(f"   网址: {source['url']}")
        print(f"   特点: {', '.join(source['features'])}")
    
    print("\n" + "=" * 80)
    print("4. 推荐下载流程")
    print("-" * 80)
    
    recommendations = """
【最推荐】CICIDS2017 数据集
  理由: 包含多种数据库攻击类型，数据真实，标注完整
  下载步骤:
    1. 访问: https://www.unb.ca/cic/datasets/ids-2017.html
    2. 注册账号并填写研究用途
    3. 下载完整数据集(约50GB)
    4. 数据包含: SQL注入、XSS、DDoS等多种攻击
  
【备选】UNSW-NB15 数据集  
  理由: 结构清晰，易于处理，包含数据库漏洞利用
  下载步骤:
    1. 访问: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    2. 直接下载CSV文件
    3. 数据包含: Fuzzers、Exploits等攻击类型
    
【快速开始】DVS_SQLi数据集
  理由: 专门针对SQL注入，数据量适中
  下载步骤:
    1. GitHub搜索: "SQL injection dataset"
    2. 选择标记数多的仓库
    3. 下载JSON或CSV格式数据
    """
    
    print(recommendations)
    
    print("\n" + "=" * 80)
    print("5. 数据增强建议")
    print("-" * 80)
    
    enhancement_tips = """
针对当前模型数据不平衡问题:

1. 数据收集策略:
   - 从CICIDS2017获取SQL注入流量样本
   - 从UNSW-NB15获取Exploits攻击样本
   - 使用WebGoat生成模拟SQL注入攻击
  
2. 数据比例调整:
   - 当前: database_attack仅占2.4%
   - 目标: 增加到至少15-20%
   - 方法: 从上述数据集中提取更多database attack样本
  
3. 数据集成:
   - 下载 → 预处理 → 标签映射 → 与现有数据合并
   - 保持特征一致性
   - 重新划分训练/测试集
  
4. 特征提取建议:
   - 重点关注SQL相关特征: 查询长度、特殊字符频率
   - 数据库协议特征: 连接数、异常查询模式
   - 响应时间异常: 数据库响应延迟
    """
    
    print(enhancement_tips)
    
    print("\n" + "=" * 80)
    print("6. 实用脚本建议")
    print("-" * 80)
    
    print("""
我可以帮你创建一个脚本来自动下载和处理数据库攻击数据:

1. download_database_attack_data.py
   - 自动从GitHub/URL下载SQL注入数据集
   - 解析和格式化数据
   - 提取数据库攻击相关特征

2. augment_dataset.py  
   - 合并新下载的数据库攻击数据
   - 重新平衡数据集
   - 生成最终的训练/测试集

3. visualize_database_attacks.py
   - 可视化数据库攻击的模式
   - 分析攻击特征分布
   - 对比不同类型攻击

需要我帮你创建这些脚本吗？
    """)
    
    print("\n" + "=" * 80)

def main():
    """主函数"""
    try:
        print_dataset_sources()
        logger.info("\n资源指南生成完成！")
        
        print("\n💡 提示:")
        print("   - 最推荐从CICIDS2017获取数据集")
        print("   - 可以同时从多个来源收集数据以提高覆盖率")
        print("   - 记得保持数据格式和特征的一致性")
        
    except Exception as e:
        logger.error(f"生成资源指南失败: {e}")
        raise

if __name__ == "__main__":
    main()
