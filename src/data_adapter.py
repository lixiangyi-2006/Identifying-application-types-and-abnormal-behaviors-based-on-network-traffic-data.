#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据适配器 - 将真实数据集转换为系统可用的格式
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RealDataAdapter:
    """真实数据适配器"""
    
    def __init__(self):
        self.feature_mapping = self._create_feature_mapping()
        self.label_mapping = {
            0: 'normal',
            1: 'brute_force',      # 暴力破解
            2: 'spoofing',         # 欺骗
            3: 'upload_attack',    # 上传危机
            4: 'database_attack'   # 数据库攻击
        }
        
    def _create_feature_mapping(self) -> Dict[str, str]:
        """创建特征映射，将原始特征名映射到系统期望的特征名"""
        return {
            # 基础流量特征
            'Flow Duration': 'flow_duration',
            'Total Fwd Packet': 'src_to_dst_packets', 
            'Total Bwd packets': 'dst_to_src_packets',
            'Total Length of Fwd Packet': 'src_to_dst_bytes',
            'Total Length of Bwd Packet': 'dst_to_src_bytes',
            
            # 数据包统计特征
            'Fwd Packet Length Max': 'max_packet_size',
            'Fwd Packet Length Min': 'min_packet_size', 
            'Fwd Packet Length Mean': 'avg_packet_size',
            'Fwd Packet Length Std': 'std_packet_size',
            'Bwd Packet Length Max': 'bwd_max_packet_size',
            'Bwd Packet Length Min': 'bwd_min_packet_size',
            'Bwd Packet Length Mean': 'bwd_avg_packet_size',
            'Bwd Packet Length Std': 'bwd_std_packet_size',
            
            # 流量速率特征
            'Flow Bytes/s': 'bytes_per_second',
            'Flow Packets/s': 'packets_per_second',
            'Fwd Packets/s': 'fwd_packets_per_second',
            'Bwd Packets/s': 'bwd_packets_per_second',
            
            # 时间间隔特征
            'Flow IAT Mean': 'inter_packet_time_mean',
            'Flow IAT Std': 'inter_packet_time_std',
            'Flow IAT Max': 'inter_packet_time_max',
            'Flow IAT Min': 'inter_packet_time_min',
            'Fwd IAT Mean': 'fwd_inter_packet_time_mean',
            'Fwd IAT Std': 'fwd_inter_packet_time_std',
            'Bwd IAT Mean': 'bwd_inter_packet_time_mean',
            'Bwd IAT Std': 'bwd_inter_packet_time_std',
            
            # TCP标志特征
            'FIN Flag Count': 'fin_flag_count',
            'SYN Flag Count': 'syn_flag_count', 
            'RST Flag Count': 'rst_flag_count',
            'PSH Flag Count': 'psh_flag_count',
            'ACK Flag Count': 'ack_flag_count',
            'URG Flag Count': 'urg_flag_count',
            
            # 其他重要特征
            'Average Packet Size': 'avg_packet_size_2',
            'Packet Length Variance': 'packet_length_variance',
            'Down/Up Ratio': 'down_up_ratio',
        }
    
    def load_and_preprocess(self, train_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载并预处理真实数据集"""
        logger.info("开始加载真实数据集...")
        
        # 加载数据
        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)
        
        logger.info(f"训练集形状: {train_df.shape}")
        logger.info(f"测试集形状: {test_df.shape}")
        
        # 预处理训练集
        train_processed = self._preprocess_dataframe(train_df, is_train=True)
        
        # 预处理测试集
        test_processed = self._preprocess_dataframe(test_df, is_train=False)
        
        return train_processed, test_processed
    
    def _preprocess_dataframe(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """预处理单个数据框"""
        logger.info(f"预处理{'训练' if is_train else '测试'}集...")
        
        # 创建副本
        df_processed = df.copy()
        
        # 1. 处理缺失值
        df_processed = self._handle_missing_values(df_processed)
        
        # 2. 重命名特征
        df_processed = self._rename_features(df_processed)
        
        # 3. 处理异常值
        df_processed = self._handle_outliers(df_processed)
        
        # 4. 特征工程
        df_processed = self._create_derived_features(df_processed)
        
        # 5. 处理标签
        if 'label' in df_processed.columns:
            df_processed['anomaly_type'] = df_processed['label'].map(self.label_mapping)
            df_processed = df_processed.drop('label', axis=1)
        
        # 6. 添加缺失的系统必需特征（用默认值填充）
        df_processed = self._add_missing_features(df_processed)
        
        logger.info(f"预处理后形状: {df_processed.shape}")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 对于数值列，用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"用中位数 {median_val:.2f} 填充 {col} 的缺失值")
        
        return df
    
    def _rename_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """重命名特征"""
        # 重命名映射的特征
        df_renamed = df.rename(columns=self.feature_mapping)
        
        # 保留未映射的特征
        unmapped_cols = [col for col in df.columns if col not in self.feature_mapping and col != 'label']
        logger.info(f"保留 {len(unmapped_cols)} 个未映射的特征")
        
        return df_renamed
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'label']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 使用截断而不是删除
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建衍生特征"""
        # 总数据包数
        if 'src_to_dst_packets' in df.columns and 'dst_to_src_packets' in df.columns:
            df['total_packets'] = df['src_to_dst_packets'] + df['dst_to_src_packets']
        
        # 总字节数
        if 'src_to_dst_bytes' in df.columns and 'dst_to_src_bytes' in df.columns:
            df['total_bytes'] = df['src_to_dst_bytes'] + df['dst_to_src_bytes']
        
        # 流量持续时间（转换为秒）
        if 'flow_duration' in df.columns:
            df['flow_duration'] = df['flow_duration'] / 1000000  # 假设原始单位是微秒
        
        return df
    
    def _add_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加系统必需的缺失特征"""
        # 系统期望的特征列表
        required_features = [
            'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
            'flow_start_time', 'flow_end_time', 'tcp_flags', 'udp_length',
            'icmp_type', 'icmp_code'
        ]
        
        for feature in required_features:
            if feature not in df.columns:
                if feature in ['src_ip', 'dst_ip']:
                    df[feature] = 3232235777  # 192.168.1.1 的整数表示
                elif feature in ['src_port', 'dst_port']:
                    df[feature] = 80  # 默认端口
                elif feature == 'protocol':
                    df[feature] = 1  # TCP = 1
                elif feature in ['flow_start_time', 'flow_end_time']:
                    df[feature] = 0.0  # 默认时间
                elif feature in ['tcp_flags', 'udp_length', 'icmp_type', 'icmp_code']:
                    df[feature] = 0  # 默认值
                logger.info(f"添加缺失特征: {feature}")
        
        # 确保所有列都是数值类型
        for col in df.columns:
            if col != 'anomaly_type' and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0)
                except:
                    # 如果转换失败，使用标签编码
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """获取特征列名（排除标签列）"""
        feature_cols = [col for col in df.columns if col not in ['anomaly_type', 'label']]
        return feature_cols

def main():
    """测试数据适配器"""
    adapter = RealDataAdapter()
    
    # 加载数据
    train_df, test_df = adapter.load_and_preprocess("data/train.xlsx", "data/test.xlsx")
    
    print("=== 数据适配器测试结果 ===")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"特征列数: {len(adapter.get_feature_columns(train_df))}")
    print(f"标签分布:")
    print(train_df['anomaly_type'].value_counts())
    
    print("\n前5行数据:")
    print(train_df.head())

if __name__ == "__main__":
    main()
