from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch

class BaseScreener(ABC):
    """
    防御/筛选器基类 (Sanitizer/Screener).
    职责：识别并剔除恶意更新，或者对更新进行降权/修正。
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: 具体的防御参数 (如 krum 的 k, trim 的 ratio)
        """
        pass

    @abstractmethod
    def screen(self, updates: List[Dict[str, Any]], server_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        核心筛选逻辑。
        
        Args:
            updates: 客户端上传的 payload 列表。
                     每个元素是一个 dict，至少包含 {'weights': ..., 'num_samples': ...}
            server_context: (可选) 服务器上下文，包含 global_model, val_loader 等信息，
                            供 FLTrust / BackdoorIndicator 等需要服务器端验证的防御使用。
                            
        Returns:
            filtered_updates: 筛选后的更新列表。
                              可以是原列表的子集，也可以是修改权重后的列表。
        """
        pass

    def __call__(self, updates, server_context=None):
        """使得 Screener 可以像函数一样被调用"""
        return self.screen(updates, server_context)