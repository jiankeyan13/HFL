import torch
from typing import List, Dict, Optional
from .base_aggregator import BaseAggregator
from core.utils.registry import AGGREGATOR_REGISTRY
@AGGREGATOR_REGISTRY.register("avg")
class AvgAggregator(BaseAggregator):
    """
    通用线性聚合器 (Linear Aggregator)。
    优化点：采用向量化运算 (Vectorization) 代替客户端循环，消除 PCIe 传输瓶颈。
    """

    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  sample_weights: Optional[List[float]] = None,
                  screen_scores: Optional[List[float]] = None,
                  global_model: torch.nn.Module = None,
                  **kwargs) -> Dict[str, torch.Tensor]:
        """
        聚合客户端 Deltas 并返回完整模型权重。
        
        Args:
            updates: 客户端模型差值列表
            sample_weights: 样本数权重列表，为 None 时执行算术平均
            screen_scores: 筛选器返回的分数列表 (0-1)，为 None 时视为全 1.0
            global_model: 全局模型
        Returns:
            完整的模型 state_dict (global_model + aggregated_delta)
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
        num_clients = len(updates)

        # 融合 sample_weights 和 screen_scores
        if sample_weights is None:
            sample_weights = [1.0] * num_clients
        if screen_scores is None:
            screen_scores = [1.0] * num_clients
            
        # 逐元素相乘得到最终权重
        final_weights = [s * sc for s, sc in zip(sample_weights, screen_scores)]
        
        # 归一化
        self._check_inputs(updates, final_weights)
        norm_weights = self._normalize_weights(final_weights)

        w_tensor = torch.tensor(norm_weights, dtype=torch.float32, device=self.device)

        aggregated_deltas = {}
        # 逐层遍历 (Layer-wise)
        layer_names = updates[0].keys() # 获取第一层的名称作为模板

        for name in layer_names:
            # 在 CPU 上堆叠所有客户端的这一层参数，产生 (num_clients, ...) 的形状
            # 然后一次性推送到目标设备 (GPU)，极大减少 PCIe 握手次数
            layer_stack = torch.stack([u[name].to(torch.float32) for u in updates]).to(self.device)

            # 加权求和
            # 将 w_tensor 的形状从 (num_clients,) 调整为 (num_clients, 1, 1, 1...)
            # 以匹配当前层 (layer_stack) 的维度
            w_view_shape = [num_clients] + [1] * (layer_stack.dim() - 1)
            w_view = w_tensor.view(*w_view_shape)

            aggregated_deltas[name] = torch.sum(layer_stack * w_view, dim=0)

        # 构建完整的 state_dict: Base + Delta
        final_weights = {}
        global_state = global_model.state_dict()
        
        for key, value in global_state.items():
            final_weights[key] = value.clone()
            if key in aggregated_deltas:
                delta = aggregated_deltas[key].to(value.device)
                final_weights[key] += delta
                    
        return final_weights