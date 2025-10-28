#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动下载数据库攻击数据集脚本
"""
import os
import requests
import zipfile
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, save_path):
    """下载文件"""
    try:
        logger.info(f"开始下载: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"下载完成: {save_path}")
        return True
    except Exception as e:
        logger.error(f"下载失败 {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """解压zip文件"""
    try:
        logger.info(f"解压文件: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"解压完成: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"解压失败: {e}")
        return False

def download_database_attack_datasets():
    """下载数据库攻击相关数据集"""
    target_dir = r"D:\data1\database_attack_data"
    os.makedirs(target_dir, exist_ok=True)
    
    logger.info(f"目标目录: {target_dir}")
    
    # 可用的公开数据集链接
    datasets = [
        {
            'name': 'SQL Injection Dataset - Small',
            'url': 'https://raw.githubusercontent.com/acarrera/ids/master/dataset/dos-simple.csv',
            'description': 'SQL注入攻击样本'
        },
        {
            'name': 'NSL-KDD Dataset',
            'url': 'https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt',
            'description': '包含数据库攻击的基准数据集'
        }
    ]
    
    print("\n尝试下载可用数据集...")
    print("注意: 大多数数据集需要手动注册下载")
    
    downloaded = []
    failed = []
    
    for dataset in datasets:
        print(f"\n处理: {dataset['name']}")
        print(f"  描述: {dataset['description']}")
        print(f"  网址: {dataset['url']}")
        
        filename = os.path.basename(dataset['url'])
        save_path = os.path.join(target_dir, filename)
        
        if download_file(dataset['url'], save_path):
            downloaded.append(dataset['name'])
            
            # 尝试解压
            if filename.endswith('.zip'):
                extract_dir = os.path.join(target_dir, filename.replace('.zip', ''))
                extract_zip(save_path, extract_dir)
        else:
            failed.append(dataset['name'])
    
    print("\n" + "=" * 70)
    print("下载结果")
    print("=" * 70)
    
    if downloaded:
        print(f"\n成功下载 ({len(downloaded)}):")
        for name in downloaded:
            print(f"  ✓ {name}")
    
    if failed:
        print(f"\n失败 ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")
    
    print("\n" + "=" * 70)
    print("手动下载建议")
    print("=" * 70)
    
    manual_downloads = """
由于大多数学术数据集需要注册，建议手动下载以下数据集:

1. CICIDS2017 (强烈推荐)
   URL: https://www.unb.ca/cic/datasets/ids-2017.html
   Steps:
     - 访问网站注册账号
     - 填写研究用途申请
     - 下载完整数据集 (约50GB)
     - 解压到 D:\\data1\\cicids2017

2. UNSW-NB15 (推荐备选)
   URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
   Steps:
     - 访问网站直接下载CSV文件
     - 包含多种数据库攻击类型
     - 保存到 D:\\data1\\unsw-nb15

3. GitHub SQL Injection Datasets
   Search: https://github.com/search?q=SQL+injection+dataset
   Steps:
     - 搜索 "SQL injection dataset"
     - 选择star数多的仓库
     - 下载JSON/CSV数据
     - 保存到 D:\\data1\\github_sql_datasets

下载后，我可以帮你整合这些数据到现有模型中。
    """
    
    print(manual_downloads)
    
    return downloaded, failed

def main():
    """主函数"""
    try:
        downloaded, failed = download_database_attack_datasets()
        
        logger.info(f"\n总共处理 {len(downloaded) + len(failed)} 个数据集")
        logger.info(f"成功: {len(downloaded)}, 失败: {len(failed)}")
        
    except Exception as e:
        logger.error(f"下载过程失败: {e}")
        raise

if __name__ == "__main__":
    main()
