import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from datetime import datetime
import glob

from shared.config import NETWORK_FEATURES, Labels
from shared.utils import setup_logging, save_dataframe, load_dataframe, validate_dataframe

class NetworkDataLoader:
    """网络流量数据加载器"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.feature_columns = NETWORK_FEATURES
        self.data_cache = {}
    
    def load_from_file(self, file_path: str, format: str = 'auto') -> pd.DataFrame:
        """从文件加载数据"""
        self.logger.info(f"从文件加载数据: {file_path}")
        
        if format == 'auto':
            format = self._detect_file_format(file_path)
        
        try:
            if format == 'parquet':
                df = pd.read_parquet(file_path)
            elif format == 'csv':
                df = pd.read_csv(file_path)
            elif format == 'json':
                df = pd.read_json(file_path)
            elif format == 'pickle':
                df = pd.read_pickle(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {format}")
            
            self.logger.info(f"数据加载成功，形状: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def load_from_directory(self, directory_path: str, pattern: str = "*.parquet") -> pd.DataFrame:
        """从目录加载多个文件"""
        self.logger.info(f"从目录加载数据: {directory_path}")
        
        # 查找匹配的文件
        file_pattern = os.path.join(directory_path, pattern)
        file_paths = glob.glob(file_pattern)
        
        if not file_paths:
            raise ValueError(f"在目录 {directory_path} 中未找到匹配 {pattern} 的文件")
        
        self.logger.info(f"找到 {len(file_paths)} 个文件")
        
        # 加载所有文件
        dataframes = []
        for file_path in file_paths:
            try:
                df = self.load_from_file(file_path)
                dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"跳过文件 {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("没有成功加载任何文件")
        
        # 合并所有数据
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"合并后数据形状: {combined_df.shape}")
        
        return combined_df
    
    def _detect_file_format(self, file_path: str) -> str:
        """自动检测文件格式"""
        _, ext = os.path.splitext(file_path.lower())
        
        format_mapping = {
            '.parquet': 'parquet',
            '.csv': 'csv',
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle'
        }
        
        return format_mapping.get(ext, 'csv')
    
    def validate_network_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证网络数据格式"""
        errors = []
        
        # 检查必需的特征列
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            errors.append(f"缺少必需特征: {missing_features}")
        
        # 检查数据类型
        numeric_features = ['flow_duration', 'total_packets', 'total_bytes', 'avg_packet_size']
        for feature in numeric_features:
            if feature in df.columns:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    errors.append(f"特征 {feature} 应该是数值类型")
        
        # 检查IP地址格式
        ip_features = ['src_ip', 'dst_ip']
        for feature in ip_features:
            if feature in df.columns:
                # 简单的IP地址格式检查
                invalid_ips = df[feature].apply(lambda x: not self._is_valid_ip(str(x)) if pd.notna(x) else False)
                if invalid_ips.any():
                    errors.append(f"特征 {feature} 包含无效IP地址")
        
        # 检查端口范围
        port_features = ['src_port', 'dst_port']
        for feature in port_features:
            if feature in df.columns:
                invalid_ports = df[feature].apply(lambda x: not (0 <= x <= 65535) if pd.notna(x) and pd.api.types.is_numeric_dtype(type(x)) else False)
                if invalid_ports.any():
                    errors.append(f"特征 {feature} 包含无效端口号")
        
        return len(errors) == 0, errors
    
    def _is_valid_ip(self, ip: str) -> bool:
        """简单的IP地址验证"""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            for part in parts:
                if not part.isdigit() or not (0 <= int(part) <= 255):
                    return False
            return True
        except:
            return False
    
    def preprocess_network_data(self, df: pd.DataFrame, 
                              target_column: Optional[str] = None) -> pd.DataFrame:
        """预处理网络数据"""
        self.logger.info("开始预处理网络数据...")
        
        df_processed = df.copy()
        
        # 1. 处理缺失值
        missing_count = df_processed.isnull().sum().sum()
        if missing_count > 0:
            self.logger.warning(f"发现 {missing_count} 个缺失值")
            
            # 数值列用中位数填充
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
                df_processed[numeric_columns].median()
            )
            
            # 分类列用众数填充
            categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                if col != target_column:  # 不处理目标列
                    mode_value = df_processed[col].mode()
                    if len(mode_value) > 0:
                        df_processed[col] = df_processed[col].fillna(mode_value[0])
        
        # 2. 数据类型转换
        # 确保数值列是数值类型
        numeric_columns = ['flow_duration', 'total_packets', 'total_bytes', 'avg_packet_size',
                          'packets_per_second', 'bytes_per_second', 'min_packet_size', 
                          'max_packet_size', 'std_packet_size']
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # 确保端口号是整数
        port_columns = ['src_port', 'dst_port']
        for col in port_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').astype('Int64')
        
        # 3. 特征工程
        df_processed = self._create_derived_features(df_processed)
        
        # 4. 异常值处理
        df_processed = self._handle_outliers(df_processed)
        
        self.logger.info(f"预处理完成，最终形状: {df_processed.shape}")
        return df_processed
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建衍生特征"""
        df_enhanced = df.copy()
        
        # 1. 流量比率特征
        if 'src_to_dst_bytes' in df_enhanced.columns and 'dst_to_src_bytes' in df_enhanced.columns:
            total_bytes = df_enhanced['src_to_dst_bytes'] + df_enhanced['dst_to_src_bytes']
            df_enhanced['src_dst_byte_ratio'] = np.where(
                total_bytes > 0,
                df_enhanced['src_to_dst_bytes'] / total_bytes,
                0.5
            )
        
        if 'src_to_dst_packets' in df_enhanced.columns and 'dst_to_src_packets' in df_enhanced.columns:
            total_packets = df_enhanced['src_to_dst_packets'] + df_enhanced['dst_to_src_packets']
            df_enhanced['src_dst_packet_ratio'] = np.where(
                total_packets > 0,
                df_enhanced['src_to_dst_packets'] / total_packets,
                0.5
            )
        
        # 2. 时间特征
        if 'flow_duration' in df_enhanced.columns:
            # 流量持续时间分类
            df_enhanced['flow_duration_category'] = pd.cut(
                df_enhanced['flow_duration'],
                bins=[0, 1, 10, 60, 3600, float('inf')],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        # 3. 包大小特征
        if 'avg_packet_size' in df_enhanced.columns:
            # 包大小分类
            df_enhanced['packet_size_category'] = pd.cut(
                df_enhanced['avg_packet_size'],
                bins=[0, 64, 512, 1024, 1500, float('inf')],
                labels=['tiny', 'small', 'medium', 'large', 'jumbo']
            )
        
        # 4. 协议特征
        if 'protocol' in df_enhanced.columns:
            # 协议类型编码
            protocol_mapping = {
                'TCP': 1, 'UDP': 2, 'ICMP': 3, 'GRE': 4, 'ESP': 5
            }
            df_enhanced['protocol_encoded'] = df_enhanced['protocol'].map(protocol_mapping).fillna(0)
        
        return df_enhanced
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        df_clean = df.copy()
        
        # 对数值特征进行异常值处理
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # 使用IQR方法检测异常值
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 将异常值替换为边界值
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str, 
                          format: str = 'parquet') -> None:
        """保存预处理后的数据"""
        self.logger.info(f"保存预处理数据到: {file_path}")
        save_dataframe(df, file_path, format)
    
    def load_processed_data(self, file_path: str, format: str = 'parquet') -> pd.DataFrame:
        """加载预处理后的数据"""
        self.logger.info(f"加载预处理数据: {file_path}")
        return load_dataframe(file_path, format)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据摘要"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        }
        
        # 目标变量分布（如果存在）
        target_columns = ['anomaly_type', 'label', 'target']
        for col in target_columns:
            if col in df.columns:
                summary[f'{col}_distribution'] = df[col].value_counts().to_dict()
                break
        
        return summary