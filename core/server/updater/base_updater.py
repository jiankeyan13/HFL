import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

class BaseUpdater:
    def __init__(self, config=None):
        pass

    def update(self, 
               global_model: torch.nn.Module, 
               aggregated_update: Dict[str, torch.Tensor], 
               calibration_loader: Optional[DataLoader] = None, 
               device: torch.device = None,
               context: Dict[str, Any] = None):
        """
        更新全局模型。
        
        Args:
            global_model: 全局模型
            aggregated_update: 聚合后的权重
            calibration_loader: BN 校准数据加载器
            device: 设备
            context: 上下文信息（如 FLAME 需要的 clip_value、noise 等）
        """
        global_model.load_state_dict(aggregated_update)
        
        if calibration_loader:
            self.calibrate_bn(global_model, calibration_loader, device)

    def calibrate_bn(self, model, loader, device):
        """代理数据BN校准"""
        model.train()
        if device: 
            model.to(device)
        with torch.no_grad():
            for data, _ in loader:
                if device: 
                    data = data.to(device)
                model(data)