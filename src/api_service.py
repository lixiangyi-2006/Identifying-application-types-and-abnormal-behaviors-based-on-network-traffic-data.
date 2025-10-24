"""
异常检测API服务
提供RESTful API接口用于模型推理
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anomaly_detection.inferencer import TwoStageAnomalyDetector
from src.anomaly_detection.data_loader import NetworkDataLoader
from shared.config import APIConfig, MODEL_DIR
from shared.utils import setup_logging

# 创建FastAPI应用
app = FastAPI(
    title=APIConfig.TITLE,
    version=APIConfig.VERSION,
    description="网络异常检测API服务"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
detector = None
data_loader = None
logger = setup_logging()

# Pydantic模型定义
class NetworkFlowData(BaseModel):
    """网络流量数据模型"""
    src_ip: str = Field(..., description="源IP地址")
    dst_ip: str = Field(..., description="目标IP地址")
    src_port: int = Field(..., description="源端口", ge=0, le=65535)
    dst_port: int = Field(..., description="目标端口", ge=0, le=65535)
    protocol: str = Field(..., description="协议类型")
    flow_duration: float = Field(..., description="流持续时间(秒)")
    total_packets: int = Field(..., description="总包数")
    total_bytes: int = Field(..., description="总字节数")
    avg_packet_size: float = Field(..., description="平均包大小")
    packets_per_second: Optional[float] = Field(None, description="每秒包数")
    bytes_per_second: Optional[float] = Field(None, description="每秒字节数")
    min_packet_size: Optional[float] = Field(None, description="最小包大小")
    max_packet_size: Optional[float] = Field(None, description="最大包大小")
    std_packet_size: Optional[float] = Field(None, description="包大小标准差")
    flow_start_time: Optional[float] = Field(None, description="流开始时间")
    flow_end_time: Optional[float] = Field(None, description="流结束时间")
    inter_packet_time_mean: Optional[float] = Field(None, description="包间时间均值")
    inter_packet_time_std: Optional[float] = Field(None, description="包间时间标准差")
    src_to_dst_packets: Optional[int] = Field(None, description="源到目标包数")
    dst_to_src_packets: Optional[int] = Field(None, description="目标到源包数")
    src_to_dst_bytes: Optional[int] = Field(None, description="源到目标字节数")
    dst_to_src_bytes: Optional[int] = Field(None, description="目标到源字节数")
    tcp_flags: Optional[int] = Field(None, description="TCP标志")
    udp_length: Optional[int] = Field(None, description="UDP长度")
    icmp_type: Optional[int] = Field(None, description="ICMP类型")
    icmp_code: Optional[int] = Field(None, description="ICMP代码")

class BatchPredictionRequest(BaseModel):
    """批量预测请求模型"""
    flows: List[NetworkFlowData] = Field(..., description="网络流量数据列表")
    confidence_threshold: float = Field(0.5, description="置信度阈值", ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    """预测响应模型"""
    prediction: int = Field(..., description="预测结果")
    label: str = Field(..., description="预测标签")
    confidence: float = Field(..., description="置信度")
    detection_path: str = Field(..., description="检测路径")
    probabilities: List[float] = Field(..., description="各类别概率")

class BatchPredictionResponse(BaseModel):
    """批量预测响应模型"""
    predictions: List[PredictionResponse] = Field(..., description="预测结果列表")
    summary: Dict[str, Any] = Field(..., description="预测摘要")

class ModelInfoResponse(BaseModel):
    """模型信息响应模型"""
    xgboost_loaded: bool = Field(..., description="XGBoost模型是否加载")
    lightgbm_loaded: bool = Field(..., description="LightGBM模型是否加载")
    xgboost_classes: List[str] = Field(..., description="XGBoost模型类别")
    feature_count: int = Field(..., description="特征数量")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    global detector, data_loader
    
    try:
        # 查找最新的XGBoost模型
        model_files = []
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.startswith("xgboost_anomaly_model_") and file.endswith(".pkl"):
                    model_files.append(os.path.join(MODEL_DIR, file))
        
        if not model_files:
            logger.warning("未找到XGBoost模型文件")
            return
        
        # 使用最新的模型
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"加载XGBoost模型: {latest_model}")
        
        # 初始化检测器和数据加载器
        detector = TwoStageAnomalyDetector(latest_model)
        data_loader = NetworkDataLoader()
        
        logger.info("模型初始化完成")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")

# API端点
@app.get("/", response_model=Dict[str, str])
async def root():
    """根端点"""
    return {
        "message": "网络异常检测API服务",
        "version": APIConfig.VERSION,
        "status": "running"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """健康检查端点"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    info = detector.get_model_info()
    return ModelInfoResponse(**info)

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(flow_data: NetworkFlowData, 
                        confidence_threshold: float = 0.5):
    """单样本预测"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 转换为DataFrame
        df = pd.DataFrame([flow_data.dict()])
        
        # 预处理数据
        df_processed = data_loader.preprocess_network_data(df)
        
        # 准备特征
        feature_columns = detector.xgboost_model.feature_columns
        if feature_columns is None:
            raise HTTPException(status_code=500, detail="模型特征列未定义")
        
        X = df_processed[feature_columns].values
        
        # 执行预测
        result = detector.predict_single(X[0], confidence_threshold)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"单样本预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """批量预测"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 转换为DataFrame
        flows_data = [flow.dict() for flow in request.flows]
        df = pd.DataFrame(flows_data)
        
        # 预处理数据
        df_processed = data_loader.preprocess_network_data(df)
        
        # 准备特征
        feature_columns = detector.xgboost_model.feature_columns
        if feature_columns is None:
            raise HTTPException(status_code=500, detail="模型特征列未定义")
        
        X = df_processed[feature_columns].values
        
        # 执行批量预测
        results = detector.batch_predict(X, request.confidence_threshold)
        
        # 构建响应
        predictions = []
        for i in range(len(request.flows)):
            prediction = PredictionResponse(
                prediction=results['predictions'][i],
                label=results['labels'][i],
                confidence=results['confidences'][i],
                detection_path=results['detection_paths'][i],
                probabilities=results['probabilities'][i]
            )
            predictions.append(prediction)
        
        # 计算摘要
        summary = {
            'total_samples': len(request.flows),
            'predicted_anomalies': sum(1 for label in results['labels'] if label != 'normal'),
            'high_confidence': sum(1 for conf in results['confidences'] if conf > 0.8),
            'needs_review': sum(1 for label in results['labels'] if label == 'needs_review')
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/file")
async def predict_from_file(file_path: str, 
                           confidence_threshold: float = 0.5,
                           background_tasks: BackgroundTasks = None):
    """从文件预测"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 加载数据
        df = data_loader.load_from_file(file_path)
        df_processed = data_loader.preprocess_network_data(df)
        
        # 准备特征
        feature_columns = detector.xgboost_model.feature_columns
        if feature_columns is None:
            raise HTTPException(status_code=500, detail="模型特征列未定义")
        
        X = df_processed[feature_columns].values
        
        # 执行预测
        results = detector.batch_predict(X, confidence_threshold)
        
        # 创建结果DataFrame
        result_df = df_processed.copy()
        result_df['predicted_label'] = results['labels']
        result_df['confidence'] = results['confidences']
        result_df['detection_path'] = results['detection_paths']
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = file_path.replace('.', f'_predictions_{timestamp}.')
        
        # 保存结果（后台任务）
        if background_tasks:
            background_tasks.add_task(
                data_loader.save_processed_data, 
                result_df, 
                output_path
            )
        
        return {
            "message": "预测完成",
            "output_path": output_path,
            "summary": {
                'total_samples': len(df),
                'predicted_anomalies': sum(1 for label in results['labels'] if label != 'normal'),
                'high_confidence': sum(1 for conf in results['confidences'] if conf > 0.8),
                'needs_review': sum(1 for label in results['labels'] if label == 'needs_review')
            }
        }
        
    except Exception as e:
        logger.error(f"文件预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_service:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=APIConfig.DEBUG
    )
