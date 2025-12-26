import torch
from typing import List, Dict, Optional, Any, Tuple
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
                  context: Dict[str, Any] = None,
                  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        聚合客户端 Deltas 并返回完整模型权重。
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
        context = context or {}
        num_clients = len(updates)

        # 融合 sample_weights 和 screen_scores
        if sample_weights is None:
            sample_weights = [1.0] * num_clients
        if screen_scores is None:
            screen_scores = [1.0] * num_clients
            
        combined_weights = [s * sc for s, sc in zip(sample_weights, screen_scores)]
        
        self._check_inputs(updates, combined_weights)
        norm_weights = self._normalize_weights(combined_weights)
        w_tensor = torch.tensor(norm_weights, dtype=torch.float32, device=self.device)

        aggregated_deltas = {}
        layer_names = updates[0].keys()

        for name in layer_names:
            layer_stack = torch.stack([u[name].to(torch.float32) for u in updates]).to(self.device)
            w_view_shape = [num_clients] + [1] * (layer_stack.dim() - 1)
            w_view = w_tensor.view(*w_view_shape)
            aggregated_deltas[name] = torch.sum(layer_stack * w_view, dim=0)

        # 构建完整的 state_dict: Base + Delta
        final_weights = {}
        global_state = global_model.state_dict()
        
        for key, value in global_state.items():
            final_weights[key] = value.clone()
            if key in aggregated_deltas:
                delta = aggregated_deltas[key].to(device=value.device, dtype=value.dtype)
                final_weights[key] += delta
                    
        return final_weights, context